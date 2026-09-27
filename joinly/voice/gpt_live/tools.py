import base64
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from joinly.types import (
    MeetingChatHistory,
    MeetingParticipant,
    Transcript,
    VideoSnapshot,
)
from joinly.voice.gpt_live.remote_tools import RemoteToolClient

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 100_000


@dataclass
class ToolResult:
    """Result of a tool call.

    Attributes:
        output: The string returned to the model as `function_call_output`.
        extra_items: Additional Responses input items to queue after the output
            (e.g. an image message for a video snapshot).
    """

    output: str
    extra_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LocalTool:
    """A tool executed in-process by joinly."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[ToolResult]]

    @property
    def schema(self) -> dict[str, Any]:
        """The Responses function-tool definition."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class MeetingActions(Protocol):
    """The subset of `MeetingSession` used by the meeting tools."""

    @property
    def transcript(self) -> Transcript: ...  # noqa: D102

    @property
    def meeting_seconds(self) -> float: ...  # noqa: D102

    async def get_chat_history(self) -> MeetingChatHistory: ...  # noqa: D102

    async def send_chat_message(self, message: str) -> None: ...  # noqa: D102

    async def get_participants(self) -> list[MeetingParticipant]: ...  # noqa: D102

    async def get_video_snapshot(self) -> VideoSnapshot: ...  # noqa: D102


def _no_params() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return "[truncated to most recent content]\n" + text[-_MAX_OUTPUT_CHARS:]


def meeting_tools(
    meeting: MeetingActions,
    *,
    flush_transcript: Callable[[], None] = lambda: None,
    on_chat_sent: Callable[[str], None] = lambda _m: None,
) -> list[LocalTool]:
    """Build the in-process meeting tools.

    Args:
        meeting: The meeting session to act on.
        flush_transcript: Called before reading the transcript so the latest
            partial segment is included.
        on_chat_sent: Called with each chat message the agent sends (so the chat
            poller can ignore the agent's own messages).
    """

    async def get_transcript(args: dict[str, Any]) -> ToolResult:
        flush_transcript()
        mode = args.get("mode", "full")
        minutes = max(0, int(args.get("minutes", 0)))
        transcript = meeting.transcript
        if mode == "latest":
            transcript = transcript.after(meeting.meeting_seconds - minutes * 60)
        elif mode == "first":
            transcript = transcript.before(minutes * 60)
        segments = [
            {
                "speaker": s.speaker or "unknown",
                "role": s.role.value,
                "text": s.text,
                "start_s": round(s.start, 1),
                "end_s": round(s.end, 1),
            }
            for s in transcript.compact().segments
        ]
        return ToolResult(
            _truncate(
                json.dumps(
                    {
                        "meeting_seconds": round(meeting.meeting_seconds),
                        "segments": segments,
                    }
                )
            )
        )

    async def get_chat_history(_args: dict[str, Any]) -> ToolResult:
        history = await meeting.get_chat_history()
        return ToolResult(_truncate(history.model_dump_json()))

    async def send_chat_message(args: dict[str, Any]) -> ToolResult:
        message = str(args["message"])
        on_chat_sent(message)
        await meeting.send_chat_message(message)
        return ToolResult("sent")

    async def get_participants(_args: dict[str, Any]) -> ToolResult:
        participants = await meeting.get_participants()
        return ToolResult(json.dumps([p.model_dump() for p in participants]))

    async def get_video_snapshot(_args: dict[str, Any]) -> ToolResult:
        snapshot = await meeting.get_video_snapshot()
        data_url = (
            f"data:{snapshot.media_type};base64,"
            f"{base64.b64encode(snapshot.data).decode()}"
        )
        image_item = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Meeting video snapshot:"},
                {"type": "input_image", "image_url": data_url, "detail": "auto"},
            ],
        }
        return ToolResult(
            "Snapshot captured; it is attached as the next user message.",
            extra_items=[image_item],
        )

    return [
        LocalTool(
            "get_transcript",
            "Get the live meeting transcript (speaker-attributed). Use mode 'latest' "
            "with minutes N for the last N minutes, 'first' for the first N minutes, "
            "or 'full' (minutes ignored) for everything.",
            {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["full", "first", "latest"]},
                    "minutes": {"type": "integer", "minimum": 0},
                },
                "required": ["mode", "minutes"],
                "additionalProperties": False,
            },
            get_transcript,
        ),
        LocalTool(
            "get_chat_history",
            "Get the messages posted in the meeting chat.",
            _no_params(),
            get_chat_history,
        ),
        LocalTool(
            "send_chat_message",
            "Post a message in the meeting chat. Use for links, lists, or anything "
            "too long to say aloud.",
            {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
            send_chat_message,
        ),
        LocalTool(
            "get_participants",
            "Get the list of meeting participants.",
            _no_params(),
            get_participants,
        ),
        LocalTool(
            "get_video_snapshot",
            "Capture what is currently visible in the meeting (webcams, shared "
            "screen). The image is attached as the next user message.",
            _no_params(),
            get_video_snapshot,
        ),
    ]


class ToolRegistry:
    """Routes function calls to local meeting tools or the remote tool server."""

    def __init__(
        self,
        local: list[LocalTool],
        remote: RemoteToolClient | None = None,
        remote_schemas: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            local: Tools executed in-process.
            remote: Client for the tool server, if configured.
            remote_schemas: Tool definitions fetched from the tool server.
            context: Meeting context sent with every remote call.
        """
        self._local = {t.name: t for t in local}
        self._remote = remote
        self._remote_schemas: dict[str, dict[str, Any]] = {}
        for schema in remote_schemas or []:
            name = schema.get("name", "")
            if name in self._local:
                logger.warning("Remote tool %r shadows a meeting tool; ignored", name)
                continue
            self._remote_schemas[name] = schema
        self.context = context or {}

    @property
    def names(self) -> list[str]:
        """All tool names."""
        return [*self._local, *self._remote_schemas]

    def schemas(self) -> list[dict[str, Any]]:
        """All tool definitions in Responses function-tool format."""
        return [t.schema for t in self._local.values()] + list(
            self._remote_schemas.values()
        )

    async def call(self, name: str, arguments: str, call_id: str) -> ToolResult:
        """Execute a function call.

        Args:
            name: The function name.
            arguments: The JSON-encoded arguments from the model.
            call_id: The model's call id (forwarded for idempotency).

        Raises:
            KeyError: If the tool is unknown.
            ValueError: If the arguments are not a JSON object.
        """
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            msg = "Tool arguments must be a JSON object"
            raise ValueError(msg)  # noqa: TRY004
        if name in self._local:
            return await self._local[name].fn(args)
        if name in self._remote_schemas and self._remote is not None:
            output = await self._remote.call(
                name, call_id=call_id, arguments=args, context=self.context
            )
            return ToolResult(output)
        msg = f"Unknown tool: {name}"
        raise KeyError(msg)
