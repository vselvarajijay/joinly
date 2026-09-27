import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from joinly.voice.gpt_live.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

type Event = dict[str, Any]

_SESSION_KEY = "_session"
_DONE_TYPES = {"response.completed", "response.done"}
_FAILED_TYPES = {"response.failed", "response.incomplete", "response.cancelled"}


@dataclass
class _ResponseState:
    key: str
    response_id: str | None = None
    tasks: dict[str, asyncio.Task[ToolResult]] = field(default_factory=dict)
    finishing: bool = False
    cancelled: bool = False


class ToolDispatcher:
    """Executes function calls of the Responses backend and continues the response.

    With Responses delegation, GPT-Live forwards backend events inside
    `response.event` envelopes. Function calls are read from nested
    `response.output_item.done` items and started immediately. When the response
    completes, one `function_call_output` is sent per call (plus any extra items,
    such as an image), followed by a single `response.create` to continue.
    """

    def __init__(  # noqa: PLR0913
        self,
        send: Callable[[Event], Awaitable[Any]],
        registry: ToolRegistry,
        *,
        tool_timeout: float = 10.0,
        on_busy: Callable[[str], None] = lambda _k: None,
        on_idle: Callable[[str], None] = lambda _k: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            send: Coroutine that sends a command to GPT-Live.
            registry: Tool registry used to execute calls.
            tool_timeout: Timeout per tool call in seconds.
            on_busy: Called with a delegation key when backend work starts.
            on_idle: Called with a delegation key when backend work has finished.
            clock: Monotonic time source in seconds.
        """
        self._send = send
        self._registry = registry
        self.tool_timeout = tool_timeout
        self._on_busy = on_busy
        self._on_idle = on_idle
        self._clock = clock
        self._states: dict[str, _ResponseState] = {}
        self._started_at: dict[str, float] = {}
        self._background: set[asyncio.Task[None]] = set()

    @property
    def active(self) -> set[str]:
        """Delegation keys with backend work in flight."""
        return set(self._started_at)

    def on_delegation_created(self, event: Event) -> None:
        """Handle `session.delegation.created`."""
        delegation = event.get("delegation", {})
        key = delegation.get("id") or _SESSION_KEY
        if delegation.get("target") not in (None, "responses"):
            logger.warning(
                "Ignoring delegation %s with target %r (only Responses delegation "
                "is supported)",
                key,
                delegation.get("target"),
            )
            return
        for other in list(self._states.values()):
            if other.key != key and other.tasks and not other.cancelled:
                logger.info("Delegation %s superseded by %s", other.key, key)
                self._cancel(other)
        self.mark_busy(key)
        logger.info("Delegation %s started", key)

    def mark_busy(self, key: str) -> None:
        """Record that backend work for `key` started (e.g. a chat request)."""
        if key not in self._started_at:
            self._started_at[key] = self._clock()
            self._on_busy(key)

    async def on_response_event(self, envelope: Event) -> None:
        """Handle a `response.event` envelope."""
        key = envelope.get("delegation_id") or _SESSION_KEY
        event = envelope.get("event", {})
        etype = event.get("type", "")

        if etype == "response.created":
            self._states[key] = _ResponseState(
                key, response_id=event.get("response", {}).get("id")
            )
            self.mark_busy(key)
        elif etype == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                self._start_call(key, item)
        elif etype in _DONE_TYPES:
            state = self._states.get(key)
            if state is not None and state.tasks:
                if not state.finishing:
                    state.finishing = True
                    task = asyncio.create_task(self._finish(state))
                    self._background.add(task)
                    task.add_done_callback(self._background.discard)
            else:
                self._states.pop(key, None)
                self._idle(key, "answered")
        elif etype in _FAILED_TYPES:
            logger.warning("Backend response %s for %s: %s", etype, key, event)
            state = self._states.pop(key, None)
            if state is not None:
                self._cancel(state)
            self._idle(key, etype)

    def expire(self, max_age: float) -> None:
        """Release delegations running longer than `max_age` seconds."""
        now = self._clock()
        for key, started in list(self._started_at.items()):
            if now - started > max_age:
                logger.warning("Delegation %s timed out after %.0fs", key, max_age)
                state = self._states.pop(key, None)
                if state is not None:
                    self._cancel(state)
                self._idle(key, "expired")

    async def aclose(self) -> None:
        """Cancel all tool calls and background work."""
        for state in self._states.values():
            self._cancel(state)
        for task in list(self._background):
            task.cancel()
        for task in list(self._background):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _start_call(self, key: str, item: Event) -> None:
        state = self._states.setdefault(key, _ResponseState(key))
        self.mark_busy(key)
        call_id = item["call_id"]
        if call_id in state.tasks:
            return
        logger.info(
            "Tool call %s(%s) [%s]", item.get("name"), item.get("arguments"), call_id
        )
        state.tasks[call_id] = asyncio.create_task(
            self._run_tool(item.get("name", ""), item.get("arguments", "{}"), call_id)
        )

    async def _run_tool(self, name: str, arguments: str, call_id: str) -> ToolResult:
        start = self._clock()
        try:
            result = await asyncio.wait_for(
                self._registry.call(name, arguments, call_id), self.tool_timeout
            )
        except TimeoutError:
            logger.warning("Tool %s timed out after %.1fs", name, self.tool_timeout)
            result = ToolResult(json.dumps({"error": f"{name} timed out"}))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Tool %s failed", name)
            result = ToolResult(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        logger.info("Tool %s finished in %.2fs", name, self._clock() - start)
        return result

    async def _finish(self, state: _ResponseState) -> None:
        results: dict[str, ToolResult] = {}
        for call_id, task in state.tasks.items():
            try:
                results[call_id] = await task
            except asyncio.CancelledError:
                if not state.cancelled:
                    raise
        if state.cancelled:
            return

        extras: list[Event] = []
        for call_id, result in results.items():
            await self._send(
                {
                    "type": "response.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result.output,
                    },
                }
            )
            extras.extend(result.extra_items)
        for item in extras:
            await self._send({"type": "response.item.create", "item": item})
        if self._states.get(state.key) is state:
            del self._states[state.key]
        await self._send({"type": "response.create"})
        logger.debug("Continued backend response for %s", state.key)

    def _cancel(self, state: _ResponseState) -> None:
        state.cancelled = True
        for task in state.tasks.values():
            task.cancel()

    def _idle(self, key: str, reason: str) -> None:
        started = self._started_at.pop(key, None)
        if started is not None:
            logger.info(
                "Delegation %s %s after %.2fs", key, reason, self._clock() - started
            )
            self._on_idle(key)
