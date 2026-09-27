from http import HTTPStatus
from typing import Any

import pytest
from fastapi.testclient import TestClient
from joinly_backend.app import create_app
from joinly_backend.registry import ToolContext, ToolRegistry

TOKEN = "secret"  # noqa: S105
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _registry() -> tuple[ToolRegistry, list[dict[str, Any]]]:
    registry = ToolRegistry()
    calls: list[dict[str, Any]] = []

    async def lookup_ticket(ticket_id: str, verbose: bool, context: ToolContext) -> str:  # noqa: FBT001
        """Look up a support ticket.

        Args:
            ticket_id: The ticket id, e.g. "SUP-1".
            verbose: Include the full history.
            context: Meeting context.
        """
        calls.append({"ticket_id": ticket_id, "context": context})
        return f"{ticket_id}: open" + (" (history)" if verbose else "")

    async def explode() -> str:
        """Always fails."""
        msg = "database down"
        raise RuntimeError(msg)

    registry.register(lookup_ticket)
    registry.register(explode)
    return registry, calls


@pytest.fixture
def client_and_calls() -> tuple[TestClient, list[dict[str, Any]]]:
    """Test client for a registry with two tools, plus recorded calls."""
    registry, calls = _registry()
    return TestClient(create_app(registry, token=TOKEN)), calls


def test_health_needs_no_auth(client_and_calls: tuple[TestClient, list]) -> None:
    """The health endpoint is public."""
    client, _ = client_and_calls
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client_and_calls: tuple[TestClient, list]) -> None:
    """Tool endpoints reject missing or wrong bearer tokens."""
    client, _ = client_and_calls
    assert client.get("/tools").status_code == HTTPStatus.UNAUTHORIZED
    assert (
        client.get("/tools", headers={"Authorization": "Bearer nope"}).status_code
        == HTTPStatus.UNAUTHORIZED
    )
    body = {"call_id": "c1", "arguments": {}}
    assert (
        client.post("/tools/explode", json=body).status_code == HTTPStatus.UNAUTHORIZED
    )


def test_schema_is_strict_and_hides_context(
    client_and_calls: tuple[TestClient, list],
) -> None:
    """Schemas are strict-mode and omit the ToolContext parameter."""
    client, _ = client_and_calls
    tools = {t["name"]: t for t in client.get("/tools", headers=AUTH).json()}
    schema = tools["lookup_ticket"]
    assert schema["type"] == "function"
    assert schema["strict"] is True
    assert schema["description"] == "Look up a support ticket."
    params = schema["parameters"]
    assert set(params["properties"]) == {"ticket_id", "verbose"}
    assert params["required"] == ["ticket_id", "verbose"]
    assert params["additionalProperties"] is False
    assert params["properties"]["ticket_id"] == {
        "type": "string",
        "description": 'The ticket id, e.g. "SUP-1".',
    }
    assert tools["explode"]["parameters"]["properties"] == {}


def test_call_round_trip_with_context(
    client_and_calls: tuple[TestClient, list],
) -> None:
    """A call runs the tool and passes the meeting context through."""
    client, calls = client_and_calls
    resp = client.post(
        "/tools/lookup_ticket",
        headers=AUTH,
        json={
            "call_id": "call_1",
            "arguments": {"ticket_id": "SUP-9", "verbose": True},
            "context": {"meeting_url": "https://teams/x", "agent_name": "Helios"},
        },
    )
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == {"output": "SUP-9: open (history)"}
    ctx = calls[0]["context"]
    assert (ctx.call_id, ctx.agent_name, ctx.meeting_url) == (
        "call_1",
        "Helios",
        "https://teams/x",
    )


def test_repeated_call_id_is_idempotent(
    client_and_calls: tuple[TestClient, list],
) -> None:
    """Retrying a call_id returns the cached output."""
    client, calls = client_and_calls
    body = {"call_id": "same", "arguments": {"ticket_id": "A", "verbose": False}}
    first = client.post("/tools/lookup_ticket", headers=AUTH, json=body).json()
    second = client.post("/tools/lookup_ticket", headers=AUTH, json=body).json()
    assert first == second
    assert len(calls) == 1


def test_error_mapping(client_and_calls: tuple[TestClient, list]) -> None:
    """Unknown tools, bad arguments and failures map to 404/422/500."""
    client, _ = client_and_calls
    unknown = client.post("/tools/nope", headers=AUTH, json={"call_id": "c"})
    assert unknown.status_code == HTTPStatus.NOT_FOUND
    assert "Unknown tool" in unknown.json()["error"]

    bad = client.post(
        "/tools/lookup_ticket",
        headers=AUTH,
        json={"call_id": "c2", "arguments": {"ticket_id": 5, "extra": 1}},
    )
    assert bad.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert "Invalid arguments" in bad.json()["error"]

    boom = client.post("/tools/explode", headers=AUTH, json={"call_id": "c3"})
    assert boom.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "database down" in boom.json()["error"]


def test_refuses_to_start_without_token() -> None:
    """The app refuses to start without a token unless in dev mode."""
    with pytest.raises(RuntimeError, match="JOINLY_BACKEND_TOKEN"):
        create_app(ToolRegistry(), token="", insecure_dev=False)
    create_app(ToolRegistry(), token="", insecure_dev=True)


def test_example_tool_registered() -> None:
    """The bundled example tool is served and works."""
    client = TestClient(create_app(token=TOKEN))
    names = [t["name"] for t in client.get("/tools", headers=AUTH).json()]
    assert "get_current_time" in names
    out = client.post(
        "/tools/get_current_time",
        headers=AUTH,
        json={"call_id": "c", "arguments": {"timezone": "UTC"}},
    ).json()
    assert "UTC" in out["output"]
