import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web

from joinly.types import (
    MeetingChatHistory,
    MeetingChatMessage,
    MeetingParticipant,
    SpeakerRole,
    Transcript,
    TranscriptSegment,
    VideoSnapshot,
)
from joinly.voice.gpt_live.connection import GptLiveConnection, GptLiveError
from joinly.voice.gpt_live.delegation import ToolDispatcher
from joinly.voice.gpt_live.remote_tools import RemoteToolClient
from joinly.voice.gpt_live.tools import (
    LocalTool,
    ToolRegistry,
    ToolResult,
    meeting_tools,
)
from tests.utils.fake_gpt_live import FakeGptLiveServer


class FakeMeeting:
    """Minimal meeting for the meeting tools."""

    def __init__(self) -> None:
        """Initialize the fake."""
        self._transcript = Transcript()
        self._transcript.add_segment(
            TranscriptSegment(
                text="Let's ship on Friday.",
                start=10,
                end=12,
                speaker="Priya",
                role=SpeakerRole.participant,
            )
        )
        self._transcript.add_segment(
            TranscriptSegment(
                text="Budget is approved.",
                start=400,
                end=402,
                speaker="Sam",
                role=SpeakerRole.participant,
            )
        )
        self.sent: list[str] = []

    @property
    def transcript(self) -> Transcript:
        """The meeting transcript."""
        return self._transcript

    @property
    def meeting_seconds(self) -> float:
        """Seconds since join."""
        return 420.0

    async def get_chat_history(self) -> MeetingChatHistory:
        """Return the fake chat history."""
        return MeetingChatHistory(
            messages=[MeetingChatMessage(text="hi", sender="Sam", timestamp="t1")]
        )

    async def send_chat_message(self, message: str) -> None:
        """Record a chat message."""
        self.sent.append(message)

    async def get_participants(self) -> list[MeetingParticipant]:
        """Return fake participants."""
        return [MeetingParticipant(name="Priya"), MeetingParticipant(name="Sam")]

    async def get_video_snapshot(self) -> VideoSnapshot:
        """Return a fake snapshot."""
        return VideoSnapshot(data=b"\xff\xd8jpeg", media_type="image/jpeg")


@pytest.fixture
async def server() -> AsyncIterator[FakeGptLiveServer]:
    """A fake GPT-Live server."""
    async with FakeGptLiveServer() as s:
        yield s


@pytest.fixture
async def tool_server() -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """A fake FastAPI-compatible tool server."""
    calls: list[dict[str, Any]] = []

    async def list_tools(request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != "Bearer secret":
            return web.json_response({"detail": "unauthorized"}, status=401)
        return web.json_response(
            [
                {
                    "type": "function",
                    "name": "lookup_ticket",
                    "description": "Look up a ticket.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticket_id": {"type": "string"}},
                        "required": ["ticket_id"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                {"type": "function", "name": "get_transcript", "parameters": {}},
            ]
        )

    async def call_tool(request: web.Request) -> web.Response:
        body = await request.json()
        calls.append({"name": request.match_info["name"], **body})
        if request.match_info["name"] == "lookup_ticket":
            return web.json_response(
                {"output": f"ticket {body['arguments']['ticket_id']}: open"}
            )
        return web.json_response({"error": "boom"}, status=500)

    app = web.Application()
    app.router.add_get("/tools", list_tools)
    app.router.add_post("/tools/{name}", call_tool)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]  # noqa: SLF001
    try:
        yield f"http://127.0.0.1:{port}", calls
    finally:
        await runner.cleanup()


def _envelope(delegation_id: str | None, event: dict[str, Any]) -> dict[str, Any]:
    return {"type": "response.event", "delegation_id": delegation_id, "event": event}


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


async def _pump(conn: GptLiveConnection, dispatcher: ToolDispatcher) -> None:
    async for event in conn.events():
        if event["type"] == "session.delegation.created":
            dispatcher.on_delegation_created(event)
        elif event["type"] == "response.event":
            await dispatcher.on_response_event(event)


async def _script_delegation(
    server: FakeGptLiveServer,
    delegation_id: str,
    calls: list[dict[str, Any]],
) -> None:
    await server.send(
        {
            "type": "session.delegation.created",
            "offset_ms": 1000,
            "delegation": {
                "id": delegation_id,
                "type": "delegation",
                "target": "responses",
            },
        }
    )
    await server.send(
        _envelope(
            delegation_id, {"type": "response.created", "response": {"id": "resp_1"}}
        )
    )
    for c in calls:
        await server.send(_envelope(delegation_id, c))
    await server.send(
        _envelope(
            delegation_id, {"type": "response.completed", "response": {"output": []}}
        )
    )


async def test_connection_start_auth_and_ack(server: FakeGptLiveServer) -> None:
    """The client sends bearer auth, starts the session and matches acks."""
    async with GptLiveConnection(server.url, "sk-test") as conn:
        session = await conn.start({"model": "gpt-live-1"})
        assert session["id"] == "sess_1"
        ack = await conn.request(
            {"type": "session.thinking.append", "delegation_id": None, "content": "x"}
        )
        assert ack["type"] == "session.thinking.appended"
    assert server.auth_headers == ["Bearer sk-test"]
    assert server.received[0]["type"] == "session.start"


async def test_connection_request_error(server: FakeGptLiveServer) -> None:
    """An `error` with a matching client_event_id rejects the request."""
    server.no_ack_types = {"session.update"}
    async with GptLiveConnection(server.url, "k") as conn:
        await conn.start({"model": "gpt-live-1"})
        request = asyncio.create_task(
            conn.request({"type": "session.update", "event_id": "upd_1", "session": {}})
        )
        await server.wait_type("session.update")
        await server.send(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "immutable_field_update",
                    "message": "nope",
                    "client_event_id": "upd_1",
                },
            }
        )
        with pytest.raises(GptLiveError, match="immutable_field_update"):
            await request


async def test_delegation_runs_local_and_remote_tools(
    server: FakeGptLiveServer, tool_server: tuple[str, list[dict[str, Any]]]
) -> None:
    """Both calls run; one response.create follows their outputs.

    Outputs are sent with matching call ids before the backend is continued.
    """
    url, remote_calls = tool_server
    meeting = FakeMeeting()
    async with (
        RemoteToolClient(url, "secret") as remote,
        GptLiveConnection(server.url, "k") as conn,
    ):
        schemas = await remote.list_tools()
        registry = ToolRegistry(
            meeting_tools(meeting), remote, schemas, context={"agent_name": "Helios"}
        )
        busy: list[str] = []
        idle: list[str] = []
        dispatcher = ToolDispatcher(
            conn.send, registry, on_busy=busy.append, on_idle=idle.append
        )
        await conn.start({"model": "gpt-live-1"})
        pump = asyncio.create_task(_pump(conn, dispatcher))

        await _script_delegation(
            server,
            "item_d1",
            [
                _call("call_a", "get_transcript", {"mode": "latest", "minutes": 5}),
                _call("call_b", "lookup_ticket", {"ticket_id": "SUP-7"}),
            ],
        )
        await server.wait_type("response.create")
        await asyncio.sleep(0.05)

        outputs = {
            e["item"]["call_id"]: e["item"]["output"]
            for e in server.of_type("response.item.create")
        }
        assert set(outputs) == {"call_a", "call_b"}
        transcript = json.loads(outputs["call_a"])
        assert [s["speaker"] for s in transcript["segments"]] == ["Sam"]  # last 5 min
        assert outputs["call_b"] == "ticket SUP-7: open"
        assert len(server.of_type("response.create")) == 1
        # outputs precede the continuation
        types = [e["type"] for e in server.received]
        assert types.index("response.create") > max(
            i for i, t in enumerate(types) if t == "response.item.create"
        )
        assert remote_calls[0]["call_id"] == "call_b"
        assert remote_calls[0]["context"] == {"agent_name": "Helios"}
        # the shadowing remote "get_transcript" was ignored
        assert registry.names.count("get_transcript") == 1

        # backend answers (no more calls) -> delegation finished
        await server.send(
            _envelope(
                "item_d1", {"type": "response.created", "response": {"id": "resp_2"}}
            )
        )
        await server.send(_envelope("item_d1", {"type": "response.completed"}))
        await asyncio.sleep(0.05)
        assert busy == ["item_d1"]
        assert idle == ["item_d1"]
        pump.cancel()


async def test_tool_errors_and_timeouts_become_error_outputs(
    server: FakeGptLiveServer,
) -> None:
    """Failures are reported to the model instead of stalling the response."""

    async def slow(_args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult("late")

    async def broken(_args: dict[str, Any]) -> ToolResult:
        msg = "kaputt"
        raise ValueError(msg)

    empty = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    registry = ToolRegistry(
        [LocalTool("slow", "", empty, slow), LocalTool("broken", "", empty, broken)]
    )
    async with GptLiveConnection(server.url, "k") as conn:
        dispatcher = ToolDispatcher(conn.send, registry, tool_timeout=0.2)
        await conn.start({"model": "gpt-live-1"})
        pump = asyncio.create_task(_pump(conn, dispatcher))
        await _script_delegation(
            server,
            "item_d1",
            [
                _call("c1", "slow", {}),
                _call("c2", "broken", {}),
                _call("c3", "missing_tool", {}),
            ],
        )
        await server.wait_type("response.create")
        outputs = {
            e["item"]["call_id"]: json.loads(e["item"]["output"])
            for e in server.of_type("response.item.create")
        }
        assert outputs["c1"] == {"error": "slow timed out"}
        assert "kaputt" in outputs["c2"]["error"]
        assert "missing_tool" in outputs["c3"]["error"]
        pump.cancel()


async def test_superseded_delegation_is_not_continued(
    server: FakeGptLiveServer,
) -> None:
    """A new delegation cancels in-flight tool calls of the previous one."""
    started = asyncio.Event()

    async def slow(_args: dict[str, Any]) -> ToolResult:
        started.set()
        await asyncio.sleep(5)
        return ToolResult("stale")

    async def fast(_args: dict[str, Any]) -> ToolResult:
        return ToolResult("fresh")

    empty = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    registry = ToolRegistry(
        [LocalTool("slow", "", empty, slow), LocalTool("fast", "", empty, fast)]
    )
    async with GptLiveConnection(server.url, "k") as conn:
        dispatcher = ToolDispatcher(conn.send, registry)
        await conn.start({"model": "gpt-live-1"})
        pump = asyncio.create_task(_pump(conn, dispatcher))
        await _script_delegation(server, "item_old", [_call("c_old", "slow", {})])
        await started.wait()
        await _script_delegation(server, "item_new", [_call("c_new", "fast", {})])
        await server.wait_type("response.create")
        await asyncio.sleep(0.1)
        call_ids = [
            e["item"]["call_id"] for e in server.of_type("response.item.create")
        ]
        assert call_ids == ["c_new"]
        assert len(server.of_type("response.create")) == 1
        pump.cancel()


async def test_snapshot_attaches_image_item(server: FakeGptLiveServer) -> None:
    """The snapshot tool returns text and queues an input_image message."""
    registry = ToolRegistry(meeting_tools(FakeMeeting()))
    async with GptLiveConnection(server.url, "k") as conn:
        dispatcher = ToolDispatcher(conn.send, registry)
        await conn.start({"model": "gpt-live-1"})
        pump = asyncio.create_task(_pump(conn, dispatcher))
        await _script_delegation(
            server, "item_d1", [_call("c1", "get_video_snapshot", {})]
        )
        await server.wait_type("response.create")
        items = [e["item"] for e in server.of_type("response.item.create")]
        assert items[0]["type"] == "function_call_output"
        image = items[1]["content"][1]
        assert image["type"] == "input_image"
        assert image["image_url"].startswith("data:image/jpeg;base64,")
        pump.cancel()


async def test_failed_response_releases_delegation(server: FakeGptLiveServer) -> None:
    """A failed backend response ends the delegation without continuing."""
    idle: list[str] = []
    async with GptLiveConnection(server.url, "k") as conn:
        dispatcher = ToolDispatcher(conn.send, ToolRegistry([]), on_idle=idle.append)
        await conn.start({"model": "gpt-live-1"})
        pump = asyncio.create_task(_pump(conn, dispatcher))
        await server.send(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "item_x", "target": "responses"},
            }
        )
        await server.send(_envelope("item_x", {"type": "response.failed"}))
        await asyncio.sleep(0.1)
        assert idle == ["item_x"]
        assert server.of_type("response.create") == []
        pump.cancel()


async def test_meeting_tools_send_chat_records_message() -> None:
    """send_chat_message posts to the meeting and notifies the callback."""
    meeting = FakeMeeting()
    sent: list[str] = []
    registry = ToolRegistry(meeting_tools(meeting, on_chat_sent=sent.append))
    result = await registry.call("send_chat_message", '{"message": "notes"}', "c1")
    assert result.output == "sent"
    assert meeting.sent == ["notes"]
    assert sent == ["notes"]
    participants = await registry.call("get_participants", "{}", "c2")
    assert [p["name"] for p in json.loads(participants.output)] == ["Priya", "Sam"]


def test_meeting_tool_schemas_are_strict() -> None:
    """All meeting tool schemas satisfy OpenAI strict-mode requirements."""
    for schema in ToolRegistry(meeting_tools(FakeMeeting())).schemas():
        params = schema["parameters"]
        assert schema["strict"] is True
        assert params["additionalProperties"] is False
        assert sorted(params["required"]) == sorted(params["properties"])
