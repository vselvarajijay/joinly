import asyncio
import base64
import contextlib
import logging
import statistics
import time
from collections.abc import Coroutine
from contextlib import AsyncExitStack
from typing import Any

from websockets.exceptions import WebSocketException

from joinly.session import MeetingSession
from joinly.types import AudioFormat, MeetingChatMessage, SpeakerRole
from joinly.utils.audio import convert_audio_format
from joinly.utils.usage import add_usage
from joinly.voice.gpt_live import prompts
from joinly.voice.gpt_live.audio import PlaybackBuffer, StreamResampler
from joinly.voice.gpt_live.bargein import SpeechActivity
from joinly.voice.gpt_live.config import GptLiveConfig
from joinly.voice.gpt_live.connection import Event, GptLiveConnection, GptLiveError
from joinly.voice.gpt_live.controllers import (
    GptLiveSpeechController,
    GptLiveTranscriptionController,
    TranscriptAccumulator,
)
from joinly.voice.gpt_live.delegation import ToolDispatcher
from joinly.voice.gpt_live.gate import AddressGate, NameMatcher, OutputRouter
from joinly.voice.gpt_live.remote_tools import RemoteToolClient, RemoteToolError
from joinly.voice.gpt_live.tools import ToolRegistry, meeting_tools

logger = logging.getLogger(__name__)

_PCM16 = 2
_SEND_INTERVAL_S = 0.04
_PLAY_CHUNK_S = 0.02
_VAD_RATES = (8000, 16000, 32000, 48000)
_RECONNECT_BACKOFF = (1, 2, 4, 8, 15, 30)
_SEED_TRANSCRIPT_CHARS = 20_000
_BARGE_IN_GRACE_S = 0.5
_PREBUFFER_S = 0.15
_STATS_INTERVAL_S = 10.0


class LatencyStats:
    """Collects response latencies (end of human speech → first audible byte)."""

    def __init__(self) -> None:
        """Initialize empty stats."""
        self.samples: dict[str, list[float]] = {"direct": [], "delegated": []}

    def add(self, kind: str, seconds: float) -> None:
        """Record a latency sample."""
        self.samples.setdefault(kind, []).append(seconds)
        logger.info("Response latency (%s): %.0f ms", kind, seconds * 1000)

    def summary(self) -> str:
        """Return a p50/p90 summary per kind."""
        parts = []
        for kind, values in self.samples.items():
            if not values:
                continue
            ordered = sorted(values)
            p90 = ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]
            parts.append(
                f"{kind}: n={len(values)} p50={statistics.median(values) * 1000:.0f}ms "
                f"p90={p90 * 1000:.0f}ms"
            )
        return "; ".join(parts) or "no samples"


class GptLiveVoiceAgent:
    """Runs a GPT-Live speech-to-speech session inside a joined meeting.

    Meeting audio is streamed to GPT-Live; its audio is played through the bot's
    virtual microphone, gated so the agent is only heard when addressed.
    Delegated backend work (Responses) is executed via the tool dispatcher.
    """

    def __init__(
        self,
        meeting: MeetingSession,
        config: GptLiveConfig,
        *,
        name: str,
        language: str,
        meeting_url: str | None = None,
    ) -> None:
        """Initialize the agent for an already joined meeting session.

        Args:
            meeting: The joined meeting session (built with voice mode gpt-live).
            config: GPT-Live configuration.
            name: The agent's name (also its meeting display name).
            language: Language code for speech.
            meeting_url: The meeting URL (forwarded to remote tools as context).
        """
        transcription = meeting.transcription_controller
        speech = meeting.speech_controller
        if not isinstance(transcription, GptLiveTranscriptionController) or not (
            isinstance(speech, GptLiveSpeechController)
        ):
            msg = "Meeting session was not built with voice mode 'gpt-live'"
            raise TypeError(msg)

        self._meeting = meeting
        self._transcription = transcription
        self._speech = speech
        self._reader = meeting.meeting_provider.audio_reader
        self._writer = meeting.meeting_provider.audio_writer
        self._cfg = config
        self.name = name
        self.language = language
        self.meeting_url = meeting_url
        self._rate = config.gpt_live_sample_rate
        self._chat_trigger = config.gpt_live_chat_trigger or f"@{name.lower()}"

        self._matcher = NameMatcher(
            name, config.gpt_live_name_aliases, self._chat_trigger
        )
        self._gate = AddressGate(followup_seconds=config.gpt_live_followup_seconds)
        self._playback = PlaybackBuffer()
        self._router = OutputRouter(
            self._gate,
            self._enqueue_playback,
            hold_seconds=config.gpt_live_hold_seconds,
            bytes_per_second=self._rate * _PCM16,
            on_speech_start=lambda: self._on_agent_utterance_start(time.monotonic()),
        )
        self._accumulator = TranscriptAccumulator(transcription.add_segment)
        self._to_writer = StreamResampler(
            self._rate, self._writer.audio_format.sample_rate
        )
        self._latency = LatencyStats()

        self._conn: GptLiveConnection | None = None
        self._dispatcher: ToolDispatcher | None = None
        self._stopping = False
        self._stop_requested = asyncio.Event()
        self._speaker: str | None = None
        self._announced_speaker: str | None = None
        self._last_speaker_note = 0.0
        self._recent_input = ""
        self._recent_output = ""
        self._sent_chat: list[str] = []
        self._last_discard_note = 0.0
        self._last_human_end: float | None = None
        self._last_latency_ref: float | None = None
        self._delegation_since_ref = False
        self._last_play_end = 0.0
        self._utterance_started = 0.0
        self._stats = {"rx": 0, "queued": 0, "written": 0}
        self._usage_seconds = 0.0
        self._background: set[asyncio.Task[Any]] = set()
        self._tasks: list[asyncio.Task[Any]] = []
        self._extra_tool_docs: list[str] = []

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        """Run until `stop()` is called."""
        async with AsyncExitStack() as stack:
            registry = await self._build_registry(stack)
            self._dispatcher = ToolDispatcher(
                self._send,
                registry,
                tool_timeout=self._cfg.backend_tool_timeout,
                on_busy=lambda key: self._on_backend_busy(key),
                on_idle=lambda key: self._gate.release(key),
            )
            stack.push_async_callback(self._dispatcher.aclose)
            self._speech.say = self.say

            self._tasks = tasks = [
                asyncio.create_task(self._mic_loop(), name="gpt-live-mic"),
                asyncio.create_task(self._player_loop(), name="gpt-live-player"),
                asyncio.create_task(self._tick_loop(), name="gpt-live-tick"),
            ]
            if self._cfg.gpt_live_chat_poll_seconds > 0:
                tasks.append(
                    asyncio.create_task(self._chat_loop(), name="gpt-live-chat")
                )
            try:
                await self._session_loop(registry)
            finally:
                self._speech.say = None
                for task in [*tasks, *self._background]:
                    task.cancel()
                for task in [*tasks, *self._background]:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                self._accumulator.flush()
                add_usage("gpt_live", {"seconds": self._usage_seconds})
                logger.info(
                    "GPT-Live finished: %.0f voice seconds; latency %s",
                    self._usage_seconds,
                    self._latency.summary(),
                )

    async def stop(self) -> None:
        """Close the GPT-Live session gracefully and end `run()`."""
        self._stopping = True
        self._stop_requested.set()
        if self._conn is not None:
            closed = await self._conn.close()
            if closed:
                logger.info("GPT-Live session closed: %s", closed.get("reason"))

    async def say(self, text: str) -> None:
        """Make the agent say `text` (used by the MCP `speak_text` tool)."""
        conn = self._require_conn()
        self._gate.open("speak_text", seconds=30)
        await conn.request(
            {
                "type": "session.instructions.append",
                "delegation_id": None,
                "content": "Immediately say the following exactly, then stop and "
                f"listen: {text}",
            }
        )

    # -------------------------------------------------------------- session

    async def _session_loop(self, registry: ToolRegistry) -> None:
        attempt = 0
        first = True
        while not self._stopping:
            try:
                async with GptLiveConnection(
                    self._cfg.gpt_live_url, self._cfg.openai_api_key.get_secret_value()
                ) as conn:
                    session = await conn.start(
                        self._session_config(registry, seed=not first)
                    )
                    self._conn = conn
                    if self._stopping:  # stop() raced with the session start
                        await conn.close()
                        break
                    self._accumulator.offset_seconds = self._transcription.now_s
                    attempt = 0
                    logger.info("GPT-Live session started: %s", session.get("id"))
                    if first:
                        first = False
                        if self._cfg.gpt_live_disclosure:
                            self._spawn(self._disclose())
                    await self._receive(conn)
            except (OSError, TimeoutError, WebSocketException, GptLiveError) as e:
                logger.warning("GPT-Live connection failed: %s", e)
            finally:
                self._conn = None
                self._accumulator.flush()
            if self._stopping:
                break
            delay = _RECONNECT_BACKOFF[min(attempt, len(_RECONNECT_BACKOFF) - 1)]
            attempt += 1
            logger.warning("Reconnecting to GPT-Live in %ds", delay)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_requested.wait(), delay)

    def _session_config(self, registry: ToolRegistry, *, seed: bool) -> dict[str, Any]:
        tools: list[dict[str, Any]] = registry.schemas()
        if self._cfg.backend_web_search:
            tools.append({"type": "web_search"})
        responses: dict[str, Any] = {
            "model": self._cfg.backend_model,
            "instructions": prompts.backend_instructions(self.name, self.language),
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": self._cfg.backend_reasoning_effort},
            "service_tier": self._cfg.backend_service_tier,
        }
        session: dict[str, Any] = {
            "model": self._cfg.gpt_live_model,
            "instructions": prompts.live_instructions(
                self.name, self.language, self._extra_tool_docs
            ),
            "audio": {
                "format": {"type": "audio/pcm", "rate": self._rate},
                "output": {"voice": self._cfg.gpt_live_voice},
            },
            "delegation": {"type": "responses", "responses": responses},
        }
        if seed:
            session["input"] = self._seed_input()
        return session

    def _seed_input(self) -> list[dict[str, Any]]:
        self._accumulator.flush()
        lines = [
            f"{s.speaker or 'unknown'}: {s.text}"
            for s in self._meeting.transcript.compact().segments
        ]
        text = "\n".join(lines)[-_SEED_TRANSCRIPT_CHARS:]
        return [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "The voice connection was restored mid-meeting. You "
                        "already introduced yourself; do not do it again. Stay silent "
                        "until addressed. Recent meeting transcript:\n" + text,
                    }
                ],
            }
        ]

    async def _receive(self, conn: GptLiveConnection) -> None:
        dispatcher = self._require_dispatcher()
        async for event in conn.events():
            etype = event.get("type", "")
            if etype == "session.output_audio.delta":
                audio = base64.b64decode(event["delta"])
                self._stats["rx"] += len(audio)
                self._router.on_audio(audio)
            elif etype == "session.input_transcript.delta":
                self._on_input_transcript(event)
            elif etype == "session.output_transcript.delta":
                self._on_output_transcript(event)
            elif etype == "session.delegation.created":
                dispatcher.on_delegation_created(event)
            elif etype == "response.event":
                await dispatcher.on_response_event(event)
            elif etype == "session.usage.updated":
                self._usage_seconds = float(event.get("usage", {}).get("seconds", 0))
            elif etype == "error":
                logger.warning("GPT-Live error: %s", event.get("error"))
            elif etype == "session.closed":
                self._usage_seconds = float(
                    event.get("usage", {}).get("seconds", self._usage_seconds)
                )
                logger.info("GPT-Live session closed (%s)", event.get("reason"))
                return

    # -------------------------------------------------------------- inputs

    def _on_input_transcript(self, event: Event) -> None:
        delta = event.get("delta", "")
        self._accumulator.add(
            SpeakerRole.participant,
            delta,
            event.get("start_ms", 0),
            event.get("end_ms", 0),
            self._speaker,
        )
        self._recent_input = (self._recent_input + delta)[-200:]
        if self._matcher.in_speech(self._recent_input):
            self._recent_input = ""
            self._gate.open("name heard")
            self._router.tick()

    def _on_output_transcript(self, event: Event) -> None:
        delta = event.get("delta", "")
        self._recent_output = (self._recent_output + delta)[-1000:]
        if self._gate.is_open:
            self._accumulator.add(
                SpeakerRole.assistant,
                delta,
                event.get("start_ms", 0),
                event.get("end_ms", 0),
                self.name,
            )

    def _on_backend_busy(self, key: str) -> None:
        self._delegation_since_ref = True
        # only keep the gate open for work started from an addressed turn; the
        # model may also delegate for conversations it was not part of
        if self._gate.is_open:
            self._gate.hold(key)

    def _on_speaker(self, speaker: str | None) -> None:
        if not speaker:
            return
        self._speaker = speaker
        now = time.monotonic()
        if (
            speaker != self._announced_speaker
            and now - self._last_speaker_note > 3.0  # noqa: PLR2004
            and self._conn is not None
        ):
            self._announced_speaker = speaker
            self._last_speaker_note = now
            self._send_nowait(
                {
                    "type": "session.thinking.append",
                    "delegation_id": None,
                    "content": f"Now speaking (per the meeting app): {speaker}",
                }
            )

    def _on_human_speech_started(self) -> None:
        # ignore talk-over right as the agent starts (echo, trailing words)
        if time.monotonic() - self._utterance_started < _BARGE_IN_GRACE_S:
            return
        # speech is audible while it is still queued (plus a short tail)
        queued_s = len(self._playback) / (self._rate * _PCM16)
        agent_audible = (
            time.monotonic() - self._router.last_speech_played < queued_s + 0.3
        )
        if agent_audible:
            dropped = self._playback.clear()
            self._router.barge_in()
            logger.info(
                "Barge-in: dropped %.2fs of agent audio",
                dropped / (self._rate * _PCM16),
            )

    # --------------------------------------------------------------- loops

    async def _mic_loop(self) -> None:
        in_fmt = self._reader.audio_format
        pcm_fmt = AudioFormat(sample_rate=in_fmt.sample_rate, byte_depth=_PCM16)
        to_live = StreamResampler(in_fmt.sample_rate, self._rate)
        vad_rate = in_fmt.sample_rate if in_fmt.sample_rate in _VAD_RATES else 16000
        to_vad = StreamResampler(in_fmt.sample_rate, vad_rate)
        activity = SpeechActivity(
            sample_rate=vad_rate, min_speech_ms=self._cfg.gpt_live_bargein_ms
        )
        batch = bytearray()
        batch_bytes = int(self._rate * _SEND_INTERVAL_S) * _PCM16
        max_offline = self._rate * _PCM16 * 3  # keep 3 s while reconnecting

        while True:
            chunk = await self._reader.read()
            self._on_speaker(chunk.speaker)
            pcm = convert_audio_format(chunk.data, in_fmt, pcm_fmt)
            started, stopped = activity.feed(to_vad.process(pcm))
            if started:
                self._on_human_speech_started()
            if stopped:
                self._last_human_end = time.monotonic()
                self._delegation_since_ref = bool(
                    self._dispatcher and self._dispatcher.active
                )
            batch.extend(to_live.process(pcm))
            conn = self._conn
            if conn is None:
                del batch[: max(0, len(batch) - max_offline)]
                continue
            if len(batch) >= batch_bytes:
                audio = base64.b64encode(bytes(batch)).decode("ascii")
                batch.clear()
                with contextlib.suppress(ConnectionError, WebSocketException):
                    await conn.send(
                        {"type": "session.input_audio.append", "audio": audio}
                    )

    async def _player_loop(self) -> None:
        """Feed buffered model audio to the virtual microphone.

        Only whole writer blocks are written: the microphone zero-pads partial
        blocks, which would splice silence into speech. Each utterance starts
        after a short jitter buffer so network gaps don't cause dropouts.
        """
        out_fmt = self._writer.audio_format
        pcm_fmt = AudioFormat(sample_rate=out_fmt.sample_rate, byte_depth=_PCM16)
        block = max(_PCM16, self._writer.chunk_size // out_fmt.byte_depth * _PCM16)
        max_bytes = int(out_fmt.sample_rate * _PLAY_CHUNK_S) * _PCM16 // block * block
        prebuffer = int(out_fmt.sample_rate * _PREBUFFER_S) * _PCM16
        while True:
            await self._playback.wait()
            now = time.monotonic()
            if now - self._last_play_end > 0.3:  # noqa: PLR2004
                # new utterance: wait for a jitter buffer (or the whole reply)
                deadline = now + _PREBUFFER_S * 2
                while (  # noqa: ASYNC110  (polls the buffer level)
                    len(self._playback) < prebuffer and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
                if not len(self._playback):
                    continue
            whole = len(self._playback) // block * block
            if whole == 0:
                # a partial block: give the next delta a moment before padding
                await asyncio.sleep(0.05)
                if len(self._playback) < block:
                    data = self._playback.pop(block)
                    if data:
                        await self._write_out(
                            convert_audio_format(data, pcm_fmt, out_fmt)
                        )
                        self._last_play_end = time.monotonic()
                continue
            data = self._playback.pop(min(whole, max_bytes))
            await self._write_out(convert_audio_format(data, pcm_fmt, out_fmt))
            self._last_play_end = time.monotonic()

    async def _tick_loop(self) -> None:
        last_stats = time.monotonic()
        while True:
            await asyncio.sleep(0.1)
            if time.monotonic() - last_stats >= _STATS_INTERVAL_S:
                last_stats = time.monotonic()
                self._log_stats()
            if self._router.tick():
                self._note_discarded()
            self._accumulator.flush_stale()
            if self._dispatcher is not None:
                self._dispatcher.expire(60)

    async def _chat_loop(self) -> None:
        seen: set[tuple[str | None, str | None, str]] | None = None
        provider = self._meeting.meeting_provider
        while True:
            await asyncio.sleep(self._cfg.gpt_live_chat_poll_seconds)
            try:
                history = await provider.get_chat_history()
            except Exception:  # noqa: BLE001
                logger.debug("Chat poll failed", exc_info=True)
                continue
            keys = [(m.sender, m.timestamp, m.text) for m in history.messages]
            if seen is None:
                seen = set(keys)
                continue
            for message, key in zip(history.messages, keys, strict=True):
                if key in seen:
                    continue
                seen.add(key)
                if self._is_own_message(message):
                    continue
                if self._matcher.in_chat(message.text):
                    await self._on_chat_request(message)

    # ------------------------------------------------------------- actions

    async def _disclose(self) -> None:
        chat = prompts.disclosure_chat(self.name, self._chat_trigger)
        self._sent_chat.append(chat)
        try:
            await self._meeting.send_chat_message(chat)
        except Exception:
            logger.exception("Failed to post disclosure in chat")
        conn = self._require_conn()
        self._recent_output = ""
        self._gate.open("disclosure", seconds=20)
        await conn.request(
            {
                "type": "session.instructions.append",
                "delegation_id": None,
                "content": prompts.disclosure_instruction(self.name),
            }
        )
        await asyncio.sleep(15)
        spoken = _word_overlap(prompts.disclosure_voice(self.name), self._recent_output)
        level = logging.INFO if spoken >= 0.8 else logging.WARNING  # noqa: PLR2004
        logger.log(level, "Disclosure spoken (%.0f%% of words matched)", spoken * 100)

    async def _on_chat_request(self, message: MeetingChatMessage) -> None:
        conn = self._conn
        if conn is None:
            return
        sender = message.sender or "a participant"
        logger.info("Chat request from %s: %s", sender, message.text)
        self._gate.open("chat mention", seconds=45)
        self._delegation_since_ref = True
        await conn.send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"[Meeting chat message to {self.name} from "
                            f"{sender}] {message.text}",
                        }
                    ],
                },
            }
        )
        await conn.send({"type": "response.create"})

    def _log_stats(self) -> None:
        bps = self._rate * _PCM16
        dead = [
            f"{t.get_name()}: {t.exception()!r}"
            for t in self._tasks
            if t.done() and not t.cancelled() and t.exception() is not None
        ]
        logger.info(
            "Audio stats: received %.1fs, queued %.1fs, written to mic %.1fs, "
            "buffered %.2fs, gate %s%s",
            self._stats["rx"] / bps,
            self._stats["queued"] / bps,
            self._stats["written"] / bps,
            len(self._playback) / bps,
            "open" if self._gate.is_open else "closed",
            f", DEAD TASKS: {dead}" if dead else "",
        )

    def _note_discarded(self) -> None:
        now = time.monotonic()
        if now - self._last_discard_note < 15 or self._conn is None:  # noqa: PLR2004
            return
        self._last_discard_note = now
        self._send_nowait(
            {
                "type": "session.thinking.append",
                "delegation_id": None,
                "content": "Your last reply was not played to the meeting because "
                f"nobody addressed you. Stay silent until someone says '{self.name}'.",
            }
        )

    def _on_agent_utterance_start(self, now: float) -> None:
        self._utterance_started = now
        ref = self._last_human_end
        if ref is None or ref == self._last_latency_ref:
            return
        elapsed = now - ref
        if 0 < elapsed < 30:  # noqa: PLR2004
            self._last_latency_ref = ref
            self._latency.add(
                "delegated" if self._delegation_since_ref else "direct", elapsed
            )

    async def _write_out(self, data: bytes) -> None:
        await self._writer.write(data)
        self._stats["written"] += len(data)

    def _enqueue_playback(self, data: bytes) -> None:
        self._stats["queued"] += len(data)
        self._playback.push(self._to_writer.process(data))

    def _is_own_message(self, message: MeetingChatMessage) -> bool:
        if message.sender and message.sender.strip().lower().startswith(
            self.name.lower()
        ):
            return True
        text = message.text.strip()
        return any(text == sent.strip() for sent in self._sent_chat)

    # ------------------------------------------------------------- helpers

    async def _build_registry(self, stack: AsyncExitStack) -> ToolRegistry:
        local = meeting_tools(
            self._meeting,
            flush_transcript=self._accumulator.flush,
            on_chat_sent=self._sent_chat.append,
        )
        remote: RemoteToolClient | None = None
        schemas: list[dict[str, Any]] = []
        if self._cfg.tools_url:
            remote = await stack.enter_async_context(
                RemoteToolClient(
                    self._cfg.tools_url,
                    self._cfg.backend_token.get_secret_value(),
                    timeout=self._cfg.backend_tool_timeout,
                )
            )
            try:
                schemas = await remote.list_tools()
            except (RemoteToolError, OSError, TimeoutError) as e:
                logger.warning(
                    "Tool server %s unavailable (%s); using meeting tools only",
                    self._cfg.tools_url,
                    e,
                )
        registry = ToolRegistry(
            local,
            remote,
            schemas,
            context={
                "meeting_url": self.meeting_url,
                "agent_name": self.name,
            },
        )
        self._extra_tool_docs = [
            f"{s['name']}: {s.get('description', '')}".strip()
            for s in schemas
            if s.get("name") in registry.names
        ]
        logger.info("Backend tools: %s", ", ".join(registry.names))
        return registry

    async def _send(self, event: Event) -> None:
        await self._require_conn().send(event)

    def _send_nowait(self, event: Event) -> None:
        self._spawn(self._send(event))

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_log_task_error)

    def _require_conn(self) -> GptLiveConnection:
        if self._conn is None:
            msg = "GPT-Live session is not connected"
            raise ConnectionError(msg)
        return self._conn

    def _require_dispatcher(self) -> ToolDispatcher:
        if self._dispatcher is None:
            msg = "Dispatcher not initialized"
            raise RuntimeError(msg)
        return self._dispatcher


def _word_overlap(expected: str, actual: str) -> float:
    """Fraction of expected words present in `actual` (case-insensitive)."""
    words = [w.strip(".,!?").lower() for w in expected.split()]
    have = {w.strip(".,!?").lower() for w in actual.split()}
    return sum(w in have for w in words) / max(1, len(words))


def _log_task_error(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("Background GPT-Live task failed: %s", exc)
