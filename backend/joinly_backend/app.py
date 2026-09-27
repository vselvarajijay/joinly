import argparse
import logging
import os
import secrets
from collections import OrderedDict
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError

from joinly_backend import tools as _tools  # noqa: F401  (registers tools)
from joinly_backend.registry import ToolContext, ToolRegistry, registry

logger = logging.getLogger(__name__)

_IDEMPOTENCY_CACHE_SIZE = 1000


class ToolCallRequest(BaseModel):
    """Body of `POST /tools/{name}`."""

    call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


def create_app(  # noqa: C901
    tool_registry: ToolRegistry | None = None,
    *,
    token: str | None = None,
    insecure_dev: bool | None = None,
) -> FastAPI:
    """Create the tool server app.

    Args:
        tool_registry: Tools to serve (default: all registered tools).
        token: Bearer token (default: `JOINLY_BACKEND_TOKEN`).
        insecure_dev: Allow running without a token (default:
            `JOINLY_BACKEND_INSECURE_DEV=1`).

    Raises:
        RuntimeError: If no token is configured and insecure mode is off.
    """
    tools = tool_registry or registry
    token = token if token is not None else os.environ.get("JOINLY_BACKEND_TOKEN", "")
    if insecure_dev is None:
        insecure_dev = os.environ.get("JOINLY_BACKEND_INSECURE_DEV") == "1"
    if not token and not insecure_dev:
        msg = (
            "JOINLY_BACKEND_TOKEN is not set "
            "(set JOINLY_BACKEND_INSECURE_DEV=1 to run without auth)"
        )
        raise RuntimeError(msg)

    bearer = HTTPBearer(auto_error=False)

    def authorize(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if not token:
            return
        if creds is None or not secrets.compare_digest(creds.credentials, token):
            raise HTTPException(status_code=401, detail="Invalid or missing token")

    app = FastAPI(title="joinly-backend", docs_url=None, redoc_url=None)
    recent: OrderedDict[str, str] = OrderedDict()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools", dependencies=[Depends(authorize)])
    async def list_tools() -> list[dict[str, Any]]:
        return tools.schemas()

    @app.post("/tools/{name}", dependencies=[Depends(authorize)])
    async def call_tool(name: str, body: ToolCallRequest) -> JSONResponse:
        if name not in tools:
            return JSONResponse({"error": f"Unknown tool: {name}"}, status_code=404)
        cache_key = f"{name}:{body.call_id}"
        if cache_key in recent:
            return JSONResponse({"output": recent[cache_key]})
        context = ToolContext.model_validate({**body.context, "call_id": body.call_id})
        try:
            output = await tools.get(name).invoke(body.arguments, context)
        except ValidationError as e:
            return JSONResponse(
                {"error": f"Invalid arguments: {e.errors(include_url=False)}"},
                status_code=422,
            )
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
        recent[cache_key] = output
        while len(recent) > _IDEMPOTENCY_CACHE_SIZE:
            recent.popitem(last=False)
        return JSONResponse({"output": output})

    return app


def main() -> None:
    """Run the tool server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(description="joinly GPT-Live tool server")
    parser.add_argument(
        "--host", default=os.environ.get("JOINLY_BACKEND_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("JOINLY_BACKEND_PORT", "8080"))
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
