# joinly-backend

Tool server for joinly's GPT-Live mode. `gpt-live-1` delegates reasoning to a
Responses model (`gpt-5.6-terra` by default); when that model calls one of the
functions defined here, joinly forwards the call to this server.

This server holds **no OpenAI key and no LLM logic** — only tools.

## API

All endpoints except `/health` require `Authorization: Bearer $JOINLY_BACKEND_TOKEN`.

| Method | Path | Body | Response |
|---|---|---|---|
| `GET` | `/health` | – | `{"status": "ok"}` |
| `GET` | `/tools` | – | list of Responses function tools (`type`, `name`, `description`, `parameters`, `strict`) |
| `POST` | `/tools/{name}` | `{"call_id", "arguments": {...}, "context": {...}}` | `{"output": "<string>"}` or `{"error": "<message>"}` (404 unknown tool, 422 bad arguments, 500 tool failure) |

`call_id` is the model's function-call id. Repeated calls with the same
`call_id` return the cached output instead of running the tool again.

## Adding a tool

```python
from joinly_backend.registry import ToolContext, tool

@tool
async def lookup_ticket(ticket_id: str, context: ToolContext) -> str:
    """Look up a support ticket by id.

    Args:
        ticket_id: The ticket id, e.g. "SUP-1234".
    """
    ...
```

Type hints and the docstring become the strict JSON schema; a parameter
annotated with `ToolContext` receives the meeting context and is not exposed to
the model. Import the module in `joinly_backend/tools/__init__.py`.

Tool arguments come from a model acting on what meeting participants said:
check permissions for anything with side effects.

## Run

```bash
JOINLY_BACKEND_TOKEN=change-me uv run joinly-backend --host 127.0.0.1 --port 8080
```
