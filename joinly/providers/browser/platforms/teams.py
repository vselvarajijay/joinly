import asyncio
import contextlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, ClassVar

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from joinly.providers.browser.platforms.base import BaseBrowserPlatformController
from joinly.settings import get_settings
from joinly.types import (
    JoinBlockedError,
    LobbyTimeoutError,
    MeetingChatHistory,
    MeetingChatMessage,
    MeetingJoinError,
    MeetingParticipant,
)

logger = logging.getLogger(__name__)

_LOBBY_RE = re.compile(
    r"please wait|let you in|you.re waiting|"
    r"someone in the meeting|waiting for (?:the )?(?:organizer|host|others)",
    re.IGNORECASE,
)
_DENIED_RE = re.compile(
    r"denied access|you.ve been removed|removed you from|"
    r"request to join (?:wasn.t|was not) (?:answered|accepted)|"
    r"no one (?:responded|let you in|admitted you)|"
    r"(?:can.t|cannot) join this meeting|"
    r"(?:doesn.t|does not) allow (?:anonymous|guests|people outside)|"
    r"anonymous (?:users|join|participants) (?:can.t|cannot|isn.t|is not|are not)|"
    r"meeting has ended|meeting is over",
    re.IGNORECASE,
)
_JOIN_ON_WEB_RE = re.compile(
    r"continue on this browser|join on the web|join from (?:your|this) browser|"
    r"use the web app",
    re.IGNORECASE,
)


class TeamsBrowserPlatformController(BaseBrowserPlatformController):
    """Controller for managing Teams browser meetings."""

    url_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?:https?://)?(?:[a-z0-9-]+\.)?(?:teams\.microsoft\.com|teams\.live\.com|teams\.microsoft\.us|dod\.teams\.microsoft\.us)/"
    )

    def __init__(self) -> None:
        """Initialize the Teams browser platform controller."""
        self._state: dict[str, Any] = {}

    @property
    def active_speaker(self) -> str | None:
        """Get the name of the active speaker in the Teams meeting."""
        return self._state.get("active_speaker")

    async def join(
        self,
        page: Page,
        url: str,
        name: str,
        passcode: str | None = None,  # noqa: ARG002
    ) -> None:
        """Join the Teams meeting.

        Args:
            page: The Playwright page instance.
            url: The URL of the Teams meeting.
            name: The name of the participant.
            passcode: The passcode for the meeting (if required).
        """
        try:
            # Check if this is a gov.teams URL
            if "teams.microsoft.us" in url or "dod.teams.microsoft.us" in url:
                await self._join_gov_teams(page, url, name)
            else:
                await self._join_standard_teams(page, url, name)

            await self._wait_for_admission(page, name, get_settings().lobby_timeout)
        except MeetingJoinError:
            await self._report_page_state(page)
            raise

        await self._setup_active_speaker_observer(page)

    async def _join_standard_teams(
        self,
        page: Page,
        url: str,
        name: str,
    ) -> None:
        """Join a standard Teams meeting.

        Args:
            page: The Playwright page instance.
            url: The URL of the Teams meeting.
            name: The name of the participant.
        """
        await page.goto(url, wait_until="load", timeout=20000)

        async def _dismiss_dialog(page: Page) -> None:
            await page.click('div[role="dialog"] button', timeout=0)

        async def _click_join_on_web(page: Page) -> None:
            btn = page.get_by_role("button", name=_JOIN_ON_WEB_RE).or_(
                page.locator('[data-tid="joinOnWeb"]')
            )
            with contextlib.suppress(PlaywrightTimeoutError):
                await btn.first.click(timeout=20000)
                logger.info("Teams: chose to continue in the browser")

        dismiss_dialog = asyncio.create_task(_dismiss_dialog(page))
        join_on_web = asyncio.create_task(_click_join_on_web(page))

        try:
            name_field = page.get_by_placeholder(re.compile("name", re.IGNORECASE))
            try:
                await name_field.fill(name, timeout=20000)
            except PlaywrightTimeoutError:
                await self._raise_if_denied(page, name)
                msg = (
                    "Teams pre-join screen did not show a name field within 20s "
                    "(guest join may require sign-in for this meeting)."
                )
                raise MeetingJoinError(msg) from None

            await self._click_join(page, name)

        finally:
            for task in (dismiss_dialog, join_on_web):
                if not task.done():
                    task.cancel()

    async def _join_gov_teams(
        self,
        page: Page,
        url: str,
        name: str,
    ) -> None:
        """Join a government Teams meeting.

        Supports teams.microsoft.us or dod.teams.microsoft.us domains.

        Args:
            page: The Playwright page instance.
            url: The URL of the Teams meeting.
            name: The name of the participant.
        """
        # Government Teams may have redirects, use longer timeout
        await page.goto(url, wait_until="load", timeout=60000)

        async def _dismiss_dialog(page: Page) -> None:
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.click('div[role="dialog"] button', timeout=1000)

        async def _click_join_browser(page: Page) -> None:
            with contextlib.suppress(PlaywrightTimeoutError):
                btn_pattern = re.compile(r"join.*browser|continue.*web", re.IGNORECASE)
                join_browser_btn = page.get_by_role("button", name=btn_pattern)
                await join_browser_btn.click(timeout=1000)

        dismiss_dialog = asyncio.create_task(_dismiss_dialog(page))
        join_browser = asyncio.create_task(_click_join_browser(page))

        try:
            name_field = page.locator(
                'input[placeholder*="name" i], input[aria-label*="name" i]'
            ).first
            await name_field.fill(name, timeout=40000)

            join_btn = page.get_by_role(
                "button", name=re.compile(r"join", re.IGNORECASE)
            )
            await join_btn.click(timeout=10000)

        finally:
            for task in [dismiss_dialog, join_browser]:
                if not task.done():
                    task.cancel()

    async def leave(self, page: Page) -> None:
        """Leave the Teams meeting.

        Args:
            page: The Playwright page instance.
        """
        leave_btn = page.get_by_role("button", name=re.compile(r"leave", re.IGNORECASE))
        if not await leave_btn.is_visible():
            msg = "Leave button not found or not visible."
            raise RuntimeError(msg)
        await leave_btn.click(timeout=1000)
        await page.wait_for_timeout(500)

    async def send_chat_message(self, page: Page, message: str) -> None:
        """Send a chat message in the Teams meeting.

        Args:
            page: The Playwright page instance.
            message: The message to send.
        """
        await self._open_chat(page)

        chat_input = page.locator("div[contenteditable='true']")
        if not await chat_input.is_visible():
            msg = "Chat input not found or not visible."
            raise RuntimeError(msg)
        await chat_input.fill(message)
        await page.wait_for_timeout(500)
        await page.keyboard.press("Enter")

    async def get_chat_history(self, page: Page) -> MeetingChatHistory:
        """Get the chat history from the Teams meeting.

        Args:
            page: The Playwright page instance.

        Returns:
            MeetingChatHistory: The chat history of the meeting.
        """
        await self._open_chat(page)

        messages: list[MeetingChatMessage] = []

        chat_items = await page.locator('[data-tid="chat-pane-item"]').all()
        for el in chat_items:
            content_el = el.locator('[data-tid="chat-pane-message"]')
            if not await content_el.count():
                continue
            text = (await content_el.first.inner_text()).strip()
            ts = await el.locator("time[datetime]").first.get_attribute("datetime")
            author_locator = el.locator('[data-tid="message-author-name"]').first
            sender_text = await author_locator.text_content() or ""
            sender = sender_text.strip() or None
            messages.append(MeetingChatMessage(text=text, timestamp=ts, sender=sender))

        return MeetingChatHistory(messages=messages)

    async def get_participants(self, page: Page) -> list[MeetingParticipant]:
        """Get the list of participants in the Teams meeting.

        Args:
            page: The Playwright page instance.

        Returns:
            list[MeetingParticipant]: A list of participants in the meeting.
        """
        participants_list = page.locator('div[aria-label="Attendees"][role="tree"]')
        is_participant_list_visible = await participants_list.is_visible()

        if not is_participant_list_visible:
            participants_button = page.get_by_role(
                "button", name=re.compile(r"^people", re.IGNORECASE)
            )
            if not await participants_button.is_visible():
                msg = "Participants button not found or not visible."
                raise RuntimeError(msg)
            await participants_button.click()
            await page.wait_for_timeout(1000)
            if not await participants_list.is_visible():
                await page.wait_for_timeout(1000)

        participants: list[MeetingParticipant] = []
        for item in await participants_list.locator(
            "[data-cid='roster-participant'][aria-label]"
        ).all():
            if aria_label := await item.get_attribute("aria-label"):
                labels = aria_label.split(", ")
                name = labels[0].strip()
                infos = labels[1:] if len(labels) > 1 else []
                participants.append(MeetingParticipant(name=name, infos=infos))

        return participants

    async def mute(self, page: Page) -> None:
        """Mute the participant in the Teams meeting.

        Args:
            page: The Playwright page instance.
        """
        mute_btn = page.get_by_role("button", name=re.compile(r"^mute", re.IGNORECASE))
        if await mute_btn.is_visible():
            await mute_btn.click(timeout=1000)
        elif not await page.get_by_role(
            "button", name=re.compile(r"^unmute", re.IGNORECASE)
        ).is_visible():
            msg = "Mute button not found or not visible."
            raise RuntimeError(msg)

    async def unmute(self, page: Page) -> None:
        """Unmute the participant in the Teams meeting.

        Args:
            page: The Playwright page instance.
        """
        unmute_btn = page.get_by_role(
            "button", name=re.compile(r"^unmute", re.IGNORECASE)
        )
        if await unmute_btn.is_visible():
            await unmute_btn.click(timeout=1000)
        elif not await page.get_by_role(
            "button", name=re.compile(r"^mute", re.IGNORECASE)
        ).is_visible():
            msg = "Unmute button not found or not visible."
            raise RuntimeError(msg)

    async def share_screen(self, page: Page) -> None:
        """Start sharing screen in the Teams meeting.

        Clicks the share toolbar button.  If Teams opens a share tray
        with options (Screen, Window, …), selects the "Screen" option
        to trigger ``getDisplayMedia``.

        Args:
            page: The Playwright page instance.
        """
        share_btn = page.get_by_role(
            "button", name=re.compile(r"share\b", re.IGNORECASE)
        )
        if not await share_btn.is_visible():
            msg = "Share button not found or not visible."
            raise RuntimeError(msg)
        await share_btn.click(timeout=2000)
        await page.wait_for_timeout(1000)

        # Teams may show a share tray — select "Screen" if present.
        screen_option = page.locator(
            'button:has-text("Screen"), '
            'button:has-text("Entire screen"), '
            '[role="menuitem"]:has-text("Screen"), '
            '[aria-label*="screen" i][role="button"], '
            '[aria-label*="Screen"][role="menuitem"]'
        ).first
        try:
            await screen_option.wait_for(state="visible", timeout=3000)
            await screen_option.click(timeout=2000)
            await page.wait_for_timeout(1000)
        except PlaywrightTimeoutError:
            # No tray — share button directly triggers getDisplayMedia
            pass

    async def stop_sharing(self, page: Page) -> None:
        """Stop sharing screen in the Teams meeting.

        The Share button is a toggle — clicking it again stops sharing.

        Args:
            page: The Playwright page instance.
        """
        share_btn = page.get_by_role(
            "button",
            name=re.compile(r"(share|stop\s+(sharing|presenting))\b", re.IGNORECASE),
        )
        if not await share_btn.first.is_visible():
            msg = "Share button not found or not visible."
            raise RuntimeError(msg)
        await share_btn.first.click(timeout=2000)
        await page.wait_for_timeout(500)

    async def _set_prejoin_devices(self, page: Page) -> None:
        """Set up devices in the pre-join screen: microphone on, camera off.

        The microphone is the bot's virtual microphone. The camera is turned off
        because the Teams v2 web client hangs on "Connecting" when the call
        starts with joinly's synthetic avatar camera stream (observed 2026-09).
        Missing switches are logged and Teams' defaults are kept.
        """
        wanted = {
            "microphone": ("toggle-mute", True),
            "camera": ("toggle-video", False),
        }
        for device, (tid, on) in wanted.items():
            switch = page.locator(f'[data-tid="{tid}"]').first
            try:
                await switch.wait_for(state="attached", timeout=15000)
                if await switch.is_checked() != on:
                    await switch.set_checked(on, timeout=3000, force=True)
                logger.info(
                    "Teams pre-join: %s %s",
                    device,
                    "on" if await switch.is_checked() else "off",
                )
            except PlaywrightTimeoutError:
                logger.warning(
                    "Teams pre-join: %s switch not found; using Teams default", device
                )

    async def _click_join(self, page: Page, name: str, attempts: int = 3) -> None:
        """Prepare the pre-join screen and click "Join now" until Teams reacts.

        The Teams v2 web client re-renders the pre-join screen (with an empty
        name field) while it boots, and ignores clicks that arrive too early
        (observed 2026-09). So before every click the name and devices are
        re-applied and the button must be enabled; afterwards it must become
        disabled (joining) or disappear.
        """
        join_btn = (
            page.locator('[data-tid="prejoin-join-button"]')
            .or_(page.get_by_role("button", name=re.compile(r"^join", re.IGNORECASE)))
            .first
        )
        name_field = page.get_by_placeholder(re.compile("name", re.IGNORECASE)).first
        for attempt in range(1, attempts + 1):
            await join_btn.wait_for(state="visible", timeout=30000)
            with contextlib.suppress(PlaywrightTimeoutError):
                if (
                    await name_field.is_visible()
                    and await name_field.input_value(timeout=1000) != name
                ):
                    await name_field.fill(name, timeout=5000)
                    logger.info("Teams: (re)entered the name '%s'", name)
            await self._set_prejoin_devices(page)
            for _ in range(30):
                if await join_btn.is_enabled():
                    break
                await asyncio.sleep(0.5)
            with contextlib.suppress(PlaywrightTimeoutError):
                await join_btn.click(timeout=5000)
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    if (
                        not await join_btn.is_visible()
                        or not await join_btn.is_enabled()
                    ):
                        logger.info("Teams: join clicked (attempt %d)", attempt)
                        return
            logger.warning(
                "Teams ignored the join click (attempt %d); retrying", attempt
            )
            await asyncio.sleep(2)
        msg = f"Teams did not react to the join button after {attempts} clicks."
        raise MeetingJoinError(msg)

    async def _report_page_state(self, page: Page) -> None:
        """Log what the page shows after a failed join (for diagnosing UI changes).

        Logs the URL and visible text. If `JOINLY_DEBUG_DIR` is set, also saves
        a screenshot and the page HTML there.
        """
        with contextlib.suppress(Exception):
            text = " ".join((await page.inner_text("body", timeout=2000)).split())
            frames = [f.url for f in page.frames if f != page.main_frame]
            logger.warning(
                "Teams page at failure: url=%s frames=%s text=%r",
                page.url,
                frames,
                text[:600],
            )
        debug_dir = os.environ.get("JOINLY_DEBUG_DIR")
        if debug_dir:
            with contextlib.suppress(Exception):
                path = Path(debug_dir)
                path.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                await page.screenshot(path=path / f"teams-join-{stamp}.png")
                (path / f"teams-join-{stamp}.html").write_text(await page.content())
                logger.warning("Saved join debug files to %s", path)

    async def _raise_if_denied(self, page: Page, name: str) -> None:
        """Raise `JoinBlockedError` if Teams shows a refusal message."""
        denied = page.get_by_text(_DENIED_RE).filter(visible=True).first
        with contextlib.suppress(PlaywrightTimeoutError):
            if await denied.is_visible(timeout=500):
                text = " ".join((await denied.inner_text(timeout=1000)).split())
                msg = f"Teams refused to let '{name}' join: {text!r}"
                raise JoinBlockedError(msg)

    async def _wait_for_admission(  # noqa: C901
        self,
        page: Page,
        name: str,
        lobby_timeout: float,
        detect_timeout: float = 30,
    ) -> None:
        """Wait until the meeting is joined, handling the lobby.

        Args:
            page: The Playwright page instance.
            name: The participant name (for log messages).
            lobby_timeout: Maximum time to wait in the lobby, in seconds.
            detect_timeout: Maximum time to recognize either the meeting or the
                lobby after clicking join, in seconds.

        Raises:
            JoinBlockedError: If Teams denies or blocks the join.
            LobbyTimeoutError: If nobody admits the participant in time.
            MeetingJoinError: If neither meeting nor lobby is recognized.
        """
        leave_btn = page.get_by_role(
            "button", name=re.compile(r"leave", re.IGNORECASE)
        ).filter(visible=True)
        lobby = page.get_by_text(_LOBBY_RE).filter(visible=True).first
        prejoin_btn = page.locator('[data-tid="prejoin-join-button"]').filter(
            visible=True
        )
        loop = asyncio.get_running_loop()
        start = loop.time()
        in_lobby_since: float | None = None
        rejoins = 0

        while True:
            now = loop.time()
            if (
                now - start > 5  # noqa: PLR2004
                and await prejoin_btn.count()
                and await prejoin_btn.first.is_enabled()
            ):
                # Teams dropped back to the pre-join screen (e.g. the join
                # attempt failed, or the call dropped right after admission)
                if rejoins >= 3:  # noqa: PLR2004
                    msg = "Teams kept returning to the pre-join screen after join."
                    raise MeetingJoinError(msg)
                rejoins += 1
                logger.warning(
                    "Teams returned to the pre-join screen; joining again (%d/3)",
                    rejoins,
                )
                await self._click_join(page, name)
                start = loop.time()
                in_lobby_since = None
                continue
            if await leave_btn.first.is_visible():
                if in_lobby_since is not None:
                    logger.info(
                        "Admitted from the Teams lobby after %.0fs",
                        now - in_lobby_since,
                    )
                logger.info("Joined the Teams meeting as '%s'", name)
                return
            await self._raise_if_denied(page, name)
            if await lobby.is_visible():
                if in_lobby_since is None:
                    in_lobby_since = now
                    logger.info(
                        "Waiting in the Teams lobby for up to %.0fs to be admitted",
                        lobby_timeout,
                    )
                elif now - in_lobby_since > lobby_timeout:
                    msg = (
                        f"Nobody admitted '{name}' from the Teams lobby within "
                        f"{lobby_timeout:.0f}s."
                    )
                    raise LobbyTimeoutError(msg)
            elif in_lobby_since is None and now - start > detect_timeout:
                if await page.get_by_text("Connecting").first.is_visible():
                    msg = (
                        f"Teams stayed on 'Connecting' for {detect_timeout:.0f}s after "
                        "clicking join (media devices may have failed to start)."
                    )
                    raise MeetingJoinError(msg)
                msg = (
                    "Could not detect the Teams meeting or lobby "
                    f"{detect_timeout:.0f}s after clicking join."
                )
                raise MeetingJoinError(msg)
            if now - start > detect_timeout + lobby_timeout:
                msg = f"Timed out joining the Teams meeting as '{name}'."
                raise LobbyTimeoutError(msg)
            await asyncio.sleep(0.5)

    async def _open_chat(self, page: Page) -> None:
        """Open the chat in the Teams meeting."""
        chat_input = page.locator("div[contenteditable='true']")
        is_chat_visible = await chat_input.is_visible()

        if not is_chat_visible:
            chat_button = page.get_by_role(
                "button", name=re.compile(r"^chat", re.IGNORECASE)
            )
            if not await chat_button.is_visible():
                msg = "Chat button not found or not visible."
                raise RuntimeError(msg)
            await chat_button.click()
            await page.wait_for_timeout(1000)
            if not await chat_input.is_visible():
                await page.wait_for_timeout(2000)

    async def _setup_active_speaker_observer(self, page: Page) -> None:
        """Setup the active speaker observer for Teams."""
        await page.expose_binding(
            "report",
            lambda _, name: self._state.update({"active_speaker": name}),
        )
        await page.evaluate(
            """
            (nameArg) => {
                const emit = n => window.report(n);
                const find = () => {
                    for (
                        const t of document.querySelectorAll(
                            'div[data-tid="stage-layout"] div[role="menuitem"]'
                        )
                    ) {
                        if (!!t.querySelector(
                            'div[data-tid="voice-level-stream-outline"].vdi-frame-occlusion'
                        )) {
                            let el = t.querySelector(
                                'div[data-tid="participant-info-nametag"]'
                            );
                            if (!el) {
                                el = t.querySelector('div:not(:has(*)):not(:empty)');
                            }
                            const name = el?.textContent.trim();
                            if (name && name.length > 0 && name !== nameArg)
                                return name;
                        }
                    }
                    return null;
                };

                let last = null, cur;
                new MutationObserver(() => {
                    cur = find();
                    if (cur !== last) { last = cur; emit(cur); }
                }).observe(
                    document,
                    {
                        subtree: true,
                        childList: true,
                        attributes: true,
                        attributeFilter: ['class']
                    }
                );
                emit(find());
            }
            """,
            get_settings().name,
        )
