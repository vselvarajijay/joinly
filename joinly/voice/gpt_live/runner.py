import asyncio
import contextlib
import logging
import signal
from typing import Any

from joinly.container import SessionContainer
from joinly.session import MeetingSession
from joinly.settings import Settings
from joinly.types import JoinBlockedError, LobbyTimeoutError, MeetingJoinError
from joinly.voice.gpt_live.config import GptLiveConfig
from joinly.voice.gpt_live.loop import GptLiveVoiceAgent

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_LOBBY_TIMEOUT = 3
EXIT_JOIN_BLOCKED = 4
EXIT_JOIN_FAILED = 5

_ALONE_CHECK_SECONDS = 120


class _PollingLogFilter(logging.Filter):
    """Drops the provider's per-call success logs for the periodic chat poll."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.msg == "Successfully performed '%s'."
            and record.args in (("get_chat_history",), ("get_participants",))
        )


async def _wait_until_alone(
    meeting: MeetingSession, name: str, alone_minutes: float
) -> None:
    """Return once the agent has been alone (or the meeting unreachable) long enough."""
    alone_since: float | None = None
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(_ALONE_CHECK_SECONDS)
        try:
            participants = await meeting.meeting_provider.get_participants()
            others = [
                p for p in participants if not p.name.lower().startswith(name.lower())
            ]
            alone = not others
        except Exception:  # noqa: BLE001
            logger.debug("Participant check failed", exc_info=True)
            alone = True
        if not alone:
            alone_since = None
            continue
        alone_since = alone_since or loop.time()
        if loop.time() - alone_since >= alone_minutes * 60:
            logger.info(
                "Nobody else in the meeting for %.0f min; leaving", alone_minutes
            )
            return


async def run(
    meeting_url: str,
    settings: Settings,
    config: GptLiveConfig | None = None,
    *,
    leave_when_alone_minutes: float = 5.0,
) -> int:
    """Join a meeting and run the GPT-Live voice agent until stopped.

    Args:
        meeting_url: The meeting URL to join.
        settings: joinly settings (voice_mode must be "gpt-live").
        config: GPT-Live configuration (default: from environment).
        leave_when_alone_minutes: Leave after being alone this long (0 disables).

    Returns:
        int: Process exit code.
    """
    config = config or GptLiveConfig()
    logging.getLogger("joinly.providers.browser.meeting_provider").addFilter(
        _PollingLogFilter()
    )
    if not config.openai_api_key.get_secret_value():
        logger.error(
            "OPENAI_API_KEY is not set; it is required for --voice-mode gpt-live"
        )
        return EXIT_CONFIG

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    async with SessionContainer(settings) as meeting:
        logger.info("Joining %s as '%s'", meeting_url, settings.name)
        try:
            await meeting.join_meeting(meeting_url, settings.name)
        except LobbyTimeoutError as e:
            logger.error("JOIN FAILED (lobby timeout): %s", e)  # noqa: TRY400
            return EXIT_LOBBY_TIMEOUT
        except JoinBlockedError as e:
            logger.error("JOIN FAILED (blocked): %s", e)  # noqa: TRY400
            return EXIT_JOIN_BLOCKED
        except MeetingJoinError as e:
            logger.error("JOIN FAILED: %s", e)  # noqa: TRY400
            return EXIT_JOIN_FAILED

        with contextlib.suppress(Exception):
            await meeting.unmute()

        agent = GptLiveVoiceAgent(
            meeting,
            config,
            name=settings.name,
            language=settings.language,
            meeting_url=meeting_url,
        )
        agent_task = asyncio.create_task(agent.run(), name="gpt-live-agent")
        waiters: list[asyncio.Task[Any]] = [
            asyncio.create_task(stop.wait(), name="stop-signal")
        ]
        if leave_when_alone_minutes > 0:
            waiters.append(
                asyncio.create_task(
                    _wait_until_alone(meeting, settings.name, leave_when_alone_minutes),
                    name="alone-watch",
                )
            )
        try:
            await asyncio.wait(
                [agent_task, *waiters], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in waiters:
                waiter.cancel()
            await agent.stop()
            try:
                await asyncio.wait_for(agent_task, timeout=20)
            except TimeoutError:
                agent_task.cancel()
            except Exception:
                logger.exception("GPT-Live agent failed")
            with contextlib.suppress(Exception):
                await meeting.leave_meeting()
            logger.info("Left the meeting")

    return EXIT_OK
