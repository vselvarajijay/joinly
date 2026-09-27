import logging
from typing import Any, Self

import aiohttp

logger = logging.getLogger(__name__)


class RemoteToolError(Exception):
    """Raised when the tool server fails or rejects a call."""


class RemoteToolClient:
    """HTTP client for the FastAPI tool server (`GET /tools`, `POST /tools/{name}`)."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the tool server, e.g. `http://backend:8080`.
            token: Shared bearer token.
            timeout: Per-request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Self:
        """Open the HTTP session (kept alive for connection reuse)."""
        self._session = aiohttp.ClientSession(
            headers=self._headers, timeout=self._timeout
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            msg = "RemoteToolClient is not open"
            raise RuntimeError(msg)
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Fetch the tool definitions in Responses function-tool format."""
        async with self._require_session().get(f"{self._base_url}/tools") as resp:
            if resp.status != 200:  # noqa: PLR2004
                msg = f"GET /tools failed: HTTP {resp.status} {await resp.text()}"
                raise RemoteToolError(msg)
            tools = await resp.json()
        if not isinstance(tools, list):
            msg = "GET /tools must return a JSON list"
            raise RemoteToolError(msg)
        return tools

    async def call(
        self,
        name: str,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        """Invoke a tool and return its string output.

        Raises:
            RemoteToolError: If the server returns an error.
        """
        payload = {"call_id": call_id, "arguments": arguments, "context": context}
        async with self._require_session().post(
            f"{self._base_url}/tools/{name}", json=payload
        ) as resp:
            try:
                body = await resp.json()
            except (aiohttp.ContentTypeError, ValueError):
                body = {"error": await resp.text()}
        if resp.status != 200 or "error" in body:  # noqa: PLR2004
            detail = body.get("error") or body.get("detail") or f"HTTP {resp.status}"
            raise RemoteToolError(str(detail))
        return str(body.get("output", ""))
