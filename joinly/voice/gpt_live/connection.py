import asyncio
import contextlib
import itertools
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Self

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

type Event = dict[str, Any]

# acknowledgment event type for each command that has one
_ACKS = {
    "session.instructions.append": "session.instructions.appended",
    "session.thinking.append": "session.thinking.appended",
    "session.commentary.append": "session.commentary.appended",
    "session.input_audio.mute": "session.input_audio.muted",
    "session.input_audio.unmute": "session.input_audio.unmuted",
    "session.update": "session.updated",
}


class GptLiveError(Exception):
    """Raised when GPT-Live rejects a command or the session fails."""

    def __init__(self, error: dict[str, Any]) -> None:
        """Initialize from an `error` event payload."""
        self.error = error
        super().__init__(
            f"{error.get('code') or error.get('type')}: {error.get('message')}"
        )


class GptLiveConnection:
    """A primary GPT-Live WebSocket session.

    Commands are JSON text messages. Commands with an acknowledgment can be awaited
    with `request`; everything else the server sends is yielded from `events`.
    """

    def __init__(self, url: str, api_key: str, *, open_timeout: float = 15) -> None:
        """Initialize the connection (call `connect` or use as context manager).

        Args:
            url: The GPT-Live WebSocket URL.
            api_key: OpenAI API key used as bearer token.
            open_timeout: Timeout for opening the WebSocket in seconds.
        """
        self._url = url
        self._api_key = api_key
        self._open_timeout = open_timeout
        self._ws: ClientConnection | None = None
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future[Event]] = {}
        self._events: asyncio.Queue[Event | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self.session: Event | None = None
        self.closed_event: Event | None = None

    async def __aenter__(self) -> Self:
        """Open the WebSocket."""
        self._ws = await connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=self._open_timeout,
            max_size=None,
        )
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the WebSocket and fail pending requests."""
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        if self._ws is not None:
            await self._ws.close()
        self._fail_pending(ConnectionError("GPT-Live connection closed"))

    def next_id(self, prefix: str = "evt") -> str:
        """Return a unique client event id."""
        return f"{prefix}_{next(self._ids)}"

    async def send(self, event: Event) -> str:
        """Send a command without waiting for an acknowledgment.

        Returns:
            str: The event id used.
        """
        if self._ws is None:
            msg = "GPT-Live connection is not open"
            raise ConnectionError(msg)
        event_id = event.setdefault("event_id", self.next_id())
        if event["type"] != "session.input_audio.append":
            logger.debug("GPT-Live <- %s (%s)", event["type"], event_id)
        await self._ws.send(json.dumps(event))
        return event_id

    async def request(self, event: Event, timeout: float = 10.0) -> Event:  # noqa: ASYNC109
        """Send a command and wait for its acknowledgment or error.

        Raises:
            GptLiveError: If the server rejects the command.
            TimeoutError: If no acknowledgment arrives in time.
        """
        event_id = event.setdefault("event_id", self.next_id())
        fut: asyncio.Future[Event] = asyncio.get_running_loop().create_future()
        self._pending[event_id] = fut
        try:
            await self.send(event)
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(event_id, None)

    async def start(self, session: Event, timeout: float = 15.0) -> Event:  # noqa: ASYNC109
        """Send `session.start` and wait for `session.started`.

        Returns:
            Event: The `session` object of `session.started`.
        """
        fut: asyncio.Future[Event] = asyncio.get_running_loop().create_future()
        self._pending["__session.started"] = fut
        try:
            await self.send({"type": "session.start", "session": session})
            started = await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop("__session.started", None)
        resolved: Event = started.get("session", {})
        self.session = resolved
        return resolved

    async def close(self, timeout: float = 15.0) -> Event | None:  # noqa: ASYNC109
        """Send `session.close` and wait for `session.closed` (final usage)."""
        if self.closed_event is not None:
            return self.closed_event
        fut: asyncio.Future[Event] = asyncio.get_running_loop().create_future()
        self._pending["__session.closed"] = fut
        try:
            await self.send({"type": "session.close"})
            return await asyncio.wait_for(fut, timeout)
        except (TimeoutError, ConnectionError, ConnectionClosed):
            logger.warning("GPT-Live finalization incomplete: no session.closed")
            return None
        finally:
            self._pending.pop("__session.closed", None)

    async def events(self) -> AsyncIterator[Event]:
        """Yield server events until the connection ends."""
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def _read_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                event: Event = json.loads(raw)
                self._dispatch(event)
        except ConnectionClosed as e:
            logger.warning("GPT-Live WebSocket closed: %s", e)
        finally:
            self._fail_pending(ConnectionError("GPT-Live connection closed"))
            self._events.put_nowait(None)

    def _dispatch(self, event: Event) -> None:
        etype = event.get("type", "")
        if etype == "session.started":
            self._resolve("__session.started", event)
        elif etype == "session.closed":
            self.closed_event = event
            self._resolve("__session.closed", event)
        elif etype == "error":
            error = event.get("error", {})
            client_id = error.get("client_event_id")
            if client_id and self._reject(client_id, GptLiveError(error)):
                return
            if "__session.started" in self._pending:
                self._reject("__session.started", GptLiveError(error))
        elif etype in _ACKS.values():
            client_id = event.get("client_event_id")
            if client_id:
                self._resolve(client_id, event)
        self._events.put_nowait(event)

    def _resolve(self, key: str, event: Event) -> None:
        fut = self._pending.get(key)
        if fut is not None and not fut.done():
            fut.set_result(event)

    def _reject(self, key: str, exc: Exception) -> bool:
        fut = self._pending.get(key)
        if fut is not None and not fut.done():
            fut.set_exception(exc)
            return True
        return False

    def _fail_pending(self, exc: Exception) -> None:
        for key, fut in list(self._pending.items()):
            if not fut.done():
                if key == "__session.closed":
                    fut.set_result(self.closed_event or {})
                else:
                    fut.set_exception(exc)
