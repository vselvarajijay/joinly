import asyncio
from collections.abc import AsyncIterator

import pytest
from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import Error as PlaywrightError

from joinly.providers.browser.platforms.teams import TeamsBrowserPlatformController
from joinly.types import JoinBlockedError, LobbyTimeoutError, MeetingJoinError

LOBBY = "<div><span>Someone in the meeting should let you in soon</span></div>"
IN_MEETING = '<div><button aria-label="Leave (Ctrl+Shift+H)">Leave</button></div>'
DENIED = "<div><span>Sorry, but you were denied access to the meeting.</span></div>"
NO_ANON = (
    "<div><span>The meeting organizer doesn't allow anonymous users "
    "to join this meeting.</span></div>"
)
ENDED = "<div><h2>This meeting has ended</h2></div>"


@pytest.fixture(scope="module")
async def browser() -> AsyncIterator[Browser]:
    """Headless Chromium (skips the module if unavailable)."""
    async with async_playwright() as p:
        try:
            b = await p.chromium.launch(headless=True)
        except PlaywrightError as e:
            pytest.skip(f"Chromium not available: {e}")
        yield b
        await b.close()


@pytest.fixture
async def page(browser: Browser) -> AsyncIterator[Page]:
    """A fresh page in its own browser context."""
    ctx = await browser.new_context()
    p = await ctx.new_page()
    yield p
    await ctx.close()


async def _admission(
    page: Page, lobby_timeout: float, detect_timeout: float = 2
) -> None:
    await TeamsBrowserPlatformController()._wait_for_admission(  # noqa: SLF001
        page, "Helios", lobby_timeout, detect_timeout=detect_timeout
    )


async def test_joined_directly(page: Page) -> None:
    """The Leave button means we're in."""
    await page.set_content(IN_MEETING)
    await _admission(page, lobby_timeout=5)


async def test_admitted_from_lobby(page: Page) -> None:
    """Lobby first, then admitted before the timeout."""
    await page.set_content(LOBBY)

    async def admit() -> None:
        await asyncio.sleep(1.0)
        await page.set_content(IN_MEETING)

    admit_task = asyncio.create_task(admit())
    await _admission(page, lobby_timeout=5)
    await admit_task


async def test_lobby_detected_despite_hidden_copy(page: Page) -> None:
    """A hidden live-region copy of the lobby text must not mask the visible one."""
    await page.set_content(
        '<div style="display:none">Someone will let you in shortly.</div>'
        "<div><span>Hi, Helios. Someone will let you in shortly.</span></div>"
    )
    with pytest.raises(LobbyTimeoutError):
        await _admission(page, lobby_timeout=1)


async def test_rejoins_after_bounce_to_prejoin(page: Page) -> None:
    """If Teams drops back to pre-join, join is clicked again and the lobby follows."""
    await page.set_content(
        """
        <button data-tid="prejoin-join-button" onclick="
            this.disabled = true;
            setTimeout(() => document.body.innerHTML =
              '<span>Someone will let you in shortly.</span>' +
              '<button onclick=&quot;0&quot;>Leave</button>', 300);
        ">Join now</button>
        """
    )
    await _admission(page, lobby_timeout=5, detect_timeout=15)


async def test_lobby_timeout(page: Page) -> None:
    """Nobody admits us: LobbyTimeoutError after the configured time."""
    await page.set_content(LOBBY)
    with pytest.raises(LobbyTimeoutError, match="lobby within 1s"):
        await _admission(page, lobby_timeout=1)


@pytest.mark.parametrize(
    ("html", "fragment"),
    [
        (DENIED, "denied access"),
        (NO_ANON, "anonymous users"),
        (ENDED, "meeting has ended"),
    ],
)
async def test_join_blocked(page: Page, html: str, fragment: str) -> None:
    """Refusal screens raise JoinBlockedError with the page text."""
    await page.set_content(html)
    with pytest.raises(JoinBlockedError, match=fragment):
        await _admission(page, lobby_timeout=5)


async def test_denied_while_waiting_in_lobby(page: Page) -> None:
    """The organizer denies the request while we wait in the lobby."""
    await page.set_content(LOBBY)

    async def deny() -> None:
        await asyncio.sleep(1.0)
        await page.set_content(DENIED)

    deny_task = asyncio.create_task(deny())
    with pytest.raises(JoinBlockedError):
        await _admission(page, lobby_timeout=10)
    await deny_task


async def test_unrecognized_page(page: Page) -> None:
    """Neither meeting nor lobby detected: MeetingJoinError (not a hang)."""
    await page.set_content("<div>Loading…</div>")
    with pytest.raises(MeetingJoinError, match="Could not detect"):
        await _admission(page, lobby_timeout=5, detect_timeout=1)


async def test_prejoin_sets_mic_on_camera_off(page: Page) -> None:
    """Mic switch is turned on and camera switch off (Teams v2 checkbox markup)."""
    await page.set_content(
        """
        <input role="switch" type="checkbox" data-tid="toggle-mute" title="Mute mic">
        <input role="switch" type="checkbox" data-tid="toggle-video" checked
               title="Turn camera off">
        """
    )
    await TeamsBrowserPlatformController()._set_prejoin_devices(page)  # noqa: SLF001
    assert await page.is_checked('[data-tid="toggle-mute"]')
    assert not await page.is_checked('[data-tid="toggle-video"]')


async def test_join_click_retried_until_teams_reacts(page: Page) -> None:
    """The first click is ignored (app not ready); the retry disables the button."""
    await page.set_content(
        """
        <button data-tid="prejoin-join-button" onclick="
            window.clicks = (window.clicks || 0) + 1;
            if (window.clicks > 1) this.disabled = true;">Join now</button>
        """
    )
    await TeamsBrowserPlatformController()._click_join(page, "Helios")  # noqa: SLF001
    assert await page.evaluate("window.clicks") == 2  # noqa: PLR2004


async def test_join_refills_name_after_rerender(page: Page) -> None:
    """Pre-join re-rendered with an empty name: the name is entered again."""
    await page.set_content(
        """
        <input placeholder="Type your name" oninput="
            document.querySelector('button').disabled = !this.value">
        <button data-tid="prejoin-join-button" disabled
                onclick="this.remove()">Join now</button>
        """
    )
    await TeamsBrowserPlatformController()._click_join(page, "Helios")  # noqa: SLF001
    assert await page.input_value("input") == "Helios"
    assert await page.locator("button").count() == 0


async def test_join_click_gives_up(page: Page) -> None:
    """A join button that never reacts ends in MeetingJoinError."""
    await page.set_content('<button data-tid="prejoin-join-button">Join now</button>')
    with pytest.raises(MeetingJoinError, match="did not react"):
        await TeamsBrowserPlatformController()._click_join(  # noqa: SLF001
            page, "Helios", attempts=2
        )
