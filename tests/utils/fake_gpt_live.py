import asyncio
import json
from collections.abc import Callable
from typing import Any, Self

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

type Event = dict[str, Any]

_ACKS = {
    "session.instructions.append": "session.instructions.appended",
    "session.thinking.append": "session.thinking.appended",
    "session.commentary.append": "session.commentary.appended",
    "session.input_audio.mute": "session.input_audio.muted",
    "session.input_audio.unmute": "session.input_audio.unmuted",
    "session.update": "session.updated",
}


class FakeGptLiveServer:
    """Scriptable stand-in for the GPT-Live WebSocket endpoint.

    Records every client event, acknowledges commands like the real service,
    answers `session.start`/`session.close`, and lets tests push server events.
    """

    def __init__(self) -> None:
        """Initialize the server (use as async context manager)."""
        self.received: list[Event] = []
        self.auth_headers: list[str | None] = []
        self.connections = 0
        self.no_ack_types: set[str] = set()
        self._ws: ServerConnection | None = None
        self._server: Server | None = None
        self._changed = asyncio.Condition()
        self.url = ""

    async def __aenter__(self) -> Self:
        """Start listening on a random localhost port."""
        self._server = await serve(self._handler, "127.0.0.1", 0)
        port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/live/sessions"
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Stop the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        self._ws = ws
        self.connections += 1
        self.auth_headers.append(
            ws.request.headers.get("Authorization") if ws.request else None
        )
        try:
            async for raw in ws:
                event: Event = json.loads(raw)
                async with self._changed:
                    self.received.append(event)
                    self._changed.notify_all()
                await self._auto_reply(ws, event)
        except ConnectionClosed:
            pass

    async def _auto_reply(self, ws: ServerConnection, event: Event) -> None:
        etype = event["type"]
        if etype == "session.start":
            await ws.send(
                json.dumps(
                    {
                        "type": "session.started",
                        "event_id": "srv_started",
                        "session": {
                            "id": f"sess_{self.connections}",
                            **event["session"],
                        },
                    }
                )
            )
        elif etype in _ACKS and etype not in self.no_ack_types:
            await ws.send(
                json.dumps(
                    {
                        "type": _ACKS[etype],
                        "event_id": f"srv_{event['event_id']}",
                        "client_event_id": event["event_id"],
                        "start_ms": 0,
                        "end_ms": 0,
                    }
                )
            )
        elif etype == "session.close":
            await ws.send(
                json.dumps(
                    {
                        "type": "session.closed",
                        "reason": "close_requested",
                        "usage": {"seconds": 42},
                    }
                )
            )
            await ws.close()

    async def send(self, event: Event) -> None:
        """Push a server event to the connected client."""
        if self._ws is None:
            msg = "No client connected"
            raise RuntimeError(msg)
        await self._ws.send(json.dumps(event))

    async def drop(self) -> None:
        """Close the current connection abruptly (simulates a network loss)."""
        if self._ws is not None:
            await self._ws.close(code=1011)

    def of_type(self, etype: str) -> list[Event]:
        """All received client events of a type."""
        return [e for e in self.received if e["type"] == etype]

    async def wait_for(
        self,
        predicate: Callable[[Event], bool],
        timeout: float = 5.0,  # noqa: ASYNC109
    ) -> Event:
        """Wait until a received client event matches `predicate`."""

        def _find() -> Event | None:
            return next((e for e in self.received if predicate(e)), None)

        async with asyncio.timeout(timeout), self._changed:
            while (found := _find()) is None:
                await self._changed.wait()
            return found

    async def wait_type(self, etype: str, timeout: float = 5.0) -> Event:  # noqa: ASYNC109
        """Wait for a client event of the given type."""
        return await self.wait_for(lambda e: e["type"] == etype, timeout)
