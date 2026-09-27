import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model


class ToolContext(BaseModel):
    """Meeting context joinly sends with every tool call."""

    model_config = ConfigDict(extra="allow")

    call_id: str = ""
    meeting_url: str | None = None
    agent_name: str | None = None


type ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class RegisteredTool:
    """A tool with its argument model and JSON schema."""

    name: str
    description: str
    fn: ToolFn
    args_model: type[BaseModel]
    context_param: str | None

    @property
    def schema(self) -> dict[str, Any]:
        """The Responses function-tool definition (strict mode)."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": _strict(self.args_model.model_json_schema()),
            "strict": True,
        }

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> str:
        """Validate arguments and run the tool.

        Raises:
            pydantic.ValidationError: If the arguments do not match the schema.
        """
        args = self.args_model.model_validate(arguments).model_dump()
        if self.context_param:
            args[self.context_param] = context
        result = await self.fn(**args)
        return result if isinstance(result, str) else str(result)


def _strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a pydantic JSON schema compatible with OpenAI strict function tools."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        prop.pop("default", None)
    schema["required"] = list(schema.get("properties", {}))
    schema["additionalProperties"] = False
    return schema


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into description and arg descriptions."""
    doc = inspect.cleandoc(doc or "")
    parts = re.split(r"^\s*Args:\s*$", doc, maxsplit=1, flags=re.MULTILINE)
    description = parts[0].strip()
    params: dict[str, str] = {}
    if len(parts) > 1:
        section = re.split(r"^\s*\w+:\s*$", parts[1], maxsplit=1, flags=re.MULTILINE)[0]
        current = None
        for line in section.splitlines():
            m = re.match(r"^\s*(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
            if m:
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    return description, params


class ToolRegistry:
    """Collects tools and exposes their schemas."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, RegisteredTool] = {}

    def __contains__(self, name: str) -> bool:
        """Whether a tool with this name is registered."""
        return name in self._tools

    def get(self, name: str) -> RegisteredTool:
        """Return a registered tool.

        Raises:
            KeyError: If the tool is unknown.
        """
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        """All tool definitions."""
        return [t.schema for t in self._tools.values()]

    def register(self, fn: ToolFn, name: str | None = None) -> ToolFn:
        """Register an async function as a tool.

        Args:
            fn: An async function with type-hinted parameters and a docstring.
            name: Tool name (defaults to the function name).
        """
        if not inspect.iscoroutinefunction(fn):
            msg = f"Tool {fn.__name__} must be an async function"
            raise TypeError(msg)
        description, param_docs = _parse_docstring(fn.__doc__ or "")
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        context_param: str | None = None
        for pname, param in inspect.signature(fn).parameters.items():
            annotation = hints.get(pname, str)
            if annotation is ToolContext:
                context_param = pname
                continue
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (
                annotation,
                Field(default=default, description=param_docs.get(pname)),
            )
        args_model = create_model(
            f"{fn.__name__}_args",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        tool_name = name or fn.__name__
        self._tools[tool_name] = RegisteredTool(
            tool_name, description, fn, args_model, context_param
        )
        return fn


registry = ToolRegistry()


def tool(fn: ToolFn) -> ToolFn:
    """Register `fn` in the default registry."""
    return registry.register(fn)
