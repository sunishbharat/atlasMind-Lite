"""
tests/test_frontend_contract.py — Frontend contract tests.

Covers every surface the frontend interacts with:

  Section 1 — Pydantic models (ClientEvent, EventAck, QueryRequest)
  Section 2 — Cancel registry unit tests (register / unregister / cancel)
  Section 3 — POST /event endpoint (cancel, heartbeat, unknown id, bad input)
  Section 4 — POST /query with request_id (registration lifecycle)
  Section 5 — Full cancel flow (query in-flight + cancel event → cancelled response)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from core.client_events import (
    ClientEvent,
    ClientEventType,
    EventAck,
    cancel,
    register,
    unregister,
    _active,
)
from core.models import QueryRequest, QueryResponse
from core.jira_auth import JiraProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_result(answer: str = "ok") -> MagicMock:
    result = MagicMock()
    result.jql = None
    result.chart_spec = None
    result.answer = answer
    result.intent_fields = []
    return result


def _make_atlasmind_mock(answer: str = "ok") -> MagicMock:
    """Return a minimal AtlasMind mock whose generate_jql resolves immediately."""
    mock = MagicMock()
    mock.generate_jql = AsyncMock(return_value=(_make_llm_result(answer), None))
    mock.field_resolver = None
    mock.standard_field_ids = []
    mock.llm_client.timeout = 300
    return mock


def _make_server_meta():
    from core.models import ServerMeta
    return ServerMeta(model_name="test-model", llm_timeout=300)


def _make_jira_profile() -> JiraProfile:
    return JiraProfile(name="test", jira_url="http://jira.test", jira_type="server")


# ---------------------------------------------------------------------------
# Section 1 — Pydantic models
# ---------------------------------------------------------------------------

class TestClientEventModel:
    def test_cancel_event_valid(self):
        e = ClientEvent(event="cancel", request_id="abc-123")
        assert e.event == ClientEventType.CANCEL
        assert e.request_id == "abc-123"

    def test_heartbeat_event_valid(self):
        e = ClientEvent(event="heartbeat", request_id="abc-123")
        assert e.event == ClientEventType.HEARTBEAT

    def test_unknown_event_type_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClientEvent(event="unknown_type", request_id="abc-123")

    def test_missing_request_id_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClientEvent(event="cancel")

    def test_missing_event_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ClientEvent(request_id="abc-123")


class TestEventAckModel:
    def test_accepted_true(self):
        ack = EventAck(request_id="abc", accepted=True, detail="cancellation requested")
        assert ack.accepted is True
        assert ack.detail == "cancellation requested"

    def test_accepted_false(self):
        ack = EventAck(request_id="abc", accepted=False, detail="no active query for this request_id")
        assert ack.accepted is False

    def test_detail_defaults_empty(self):
        ack = EventAck(request_id="abc", accepted=True)
        assert ack.detail == ""

    def test_request_id_preserved(self):
        ack = EventAck(request_id="my-uuid", accepted=True)
        assert ack.request_id == "my-uuid"


class TestQueryRequestModel:
    def test_request_id_optional(self):
        req = QueryRequest(query="list bugs")
        assert req.request_id is None

    def test_request_id_accepted(self):
        req = QueryRequest(query="list bugs", request_id="uuid-xyz")
        assert req.request_id == "uuid-xyz"

    def test_limit_optional(self):
        req = QueryRequest(query="list bugs")
        assert req.limit is None

    def test_all_fields(self):
        req = QueryRequest(query="list bugs", request_id="r1", limit=50, profile="apache")
        assert req.query == "list bugs"
        assert req.request_id == "r1"
        assert req.limit == 50
        assert req.profile == "apache"


# ---------------------------------------------------------------------------
# Section 2 — Cancel registry unit tests
# ---------------------------------------------------------------------------

class TestCancelRegistry:
    def setup_method(self):
        """Clear the global registry before each test."""
        _active.clear()

    def _make_task(self) -> MagicMock:
        return MagicMock(spec=asyncio.Task)

    def test_register_adds_token(self):
        register("req-1")
        assert "req-1" in _active

    def test_register_returns_token(self):
        token = register("req-1")
        assert token is _active["req-1"]

    def test_cancel_with_attached_task(self):
        task = self._make_task()
        token = register("req-1")
        token.attach(task)
        result = cancel("req-1")
        assert result is True
        task.cancel.assert_called_once()

    def test_cancel_before_attach_sets_flag(self):
        """Cancel arriving before task is attached must fire on attach."""
        task = self._make_task()
        token = register("req-1")
        cancel("req-1")                # cancel fires before attach
        token.attach(task)             # attach must cancel immediately
        task.cancel.assert_called_once()

    def test_cancel_unknown_id_returns_false(self):
        result = cancel("no-such-id")
        assert result is False

    def test_unregister_removes_token(self):
        register("req-1")
        unregister("req-1")
        assert "req-1" not in _active

    def test_cancel_after_unregister_returns_false(self):
        register("req-1")
        unregister("req-1")
        assert cancel("req-1") is False

    def test_unregister_unknown_id_is_safe(self):
        unregister("ghost-id")

    def test_multiple_tokens_independent(self):
        t1, t2 = self._make_task(), self._make_task()
        tok1 = register("req-1")
        tok2 = register("req-2")
        tok1.attach(t1)
        tok2.attach(t2)
        cancel("req-1")
        t1.cancel.assert_called_once()
        t2.cancel.assert_not_called()

    def test_register_overwrites_existing_id(self):
        """Re-registering the same ID replaces the old token (retry scenario)."""
        t1, t2 = self._make_task(), self._make_task()
        tok1 = register("req-1")
        tok1.attach(t1)
        tok2 = register("req-1")       # overwrite
        tok2.attach(t2)
        cancel("req-1")
        t2.cancel.assert_called_once()
        t1.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# Section 3 — POST /event endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client():
    """TestClient with _atlasmind and _server_meta patched — no real DB/model."""
    from server import app
    mock_am = _make_atlasmind_mock()
    mock_meta = _make_server_meta()
    with patch("server._atlasmind", mock_am), \
         patch("server._server_meta", mock_meta), \
         patch("server.load_active_jira_profile", return_value=_make_jira_profile()):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


class TestEventEndpoint:
    def setup_method(self):
        _active.clear()

    def test_cancel_active_request_returns_accepted(self, app_client):
        task = MagicMock(spec=asyncio.Task)
        token = register("req-active")
        token.attach(task)
        resp = app_client.post("/event", json={"event": "cancel", "request_id": "req-active"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is True
        assert body["request_id"] == "req-active"
        assert "cancellation requested" in body["detail"]
        task.cancel.assert_called_once()

    def test_cancel_unknown_request_returns_not_accepted(self, app_client):
        resp = app_client.post("/event", json={"event": "cancel", "request_id": "ghost"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is False
        assert body["request_id"] == "ghost"
        assert "no active query" in body["detail"]

    def test_heartbeat_always_accepted(self, app_client):
        resp = app_client.post("/event", json={"event": "heartbeat", "request_id": "r1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] is True
        assert body["detail"] == "ok"

    def test_invalid_event_type_returns_422(self, app_client):
        resp = app_client.post("/event", json={"event": "explode", "request_id": "r1"})
        assert resp.status_code == 422

    def test_missing_request_id_returns_422(self, app_client):
        resp = app_client.post("/event", json={"event": "cancel"})
        assert resp.status_code == 422

    def test_missing_event_field_returns_422(self, app_client):
        resp = app_client.post("/event", json={"request_id": "r1"})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, app_client):
        resp = app_client.post("/event", json={})
        assert resp.status_code == 422

    def test_event_ack_shape(self, app_client):
        """Response must always contain request_id, accepted, and detail."""
        resp = app_client.post("/event", json={"event": "heartbeat", "request_id": "shape-test"})
        body = resp.json()
        assert "request_id" in body
        assert "accepted" in body
        assert "detail" in body


# ---------------------------------------------------------------------------
# Section 4 — POST /query with request_id (registration lifecycle)
# ---------------------------------------------------------------------------

class TestQueryRequestId:
    def setup_method(self):
        _active.clear()

    def test_query_without_request_id_succeeds(self, app_client):
        resp = app_client.post("/query", json={"query": "list bugs"})
        assert resp.status_code == 200

    def test_query_with_request_id_succeeds(self, app_client):
        resp = app_client.post("/query", json={"query": "list bugs", "request_id": "uuid-1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("type") in ("general", "jql")

    def test_task_unregistered_after_query_completes(self, app_client):
        """Registry must be clean after the query finishes (finally block)."""
        app_client.post("/query", json={"query": "list bugs", "request_id": "cleanup-test"})
        assert "cleanup-test" not in _active

    def test_response_has_answer(self, app_client):
        resp = app_client.post("/query", json={"query": "hello", "request_id": "r1"})
        body = resp.json()
        assert "answer" in body

    def test_get_query_without_request_id_succeeds(self, app_client):
        resp = app_client.get("/query", params={"q": "list bugs"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Section 5 — Full cancel flow (async, query in-flight + cancel event)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_mid_flight_returns_cancelled_response():
    """
    Fire a slow /query (mocked to sleep), send a cancel event while it is
    in-flight, and assert the response signals cancellation.
    """
    from server import app

    cancel_event = asyncio.Event()

    async def slow_generate_jql(query, jira_token=None, jira_url=None):
        cancel_event.set()          # signal that the query is now in-flight
        await asyncio.sleep(10)     # long enough for cancel to arrive
        return _make_llm_result("should not reach here"), None

    mock_am = MagicMock()
    mock_am.generate_jql = slow_generate_jql
    mock_am.field_resolver = None
    mock_am.standard_field_ids = []
    mock_am.llm_client.timeout = 300
    mock_meta = _make_server_meta()

    request_id = "cancel-flow-uuid"

    with patch("server._atlasmind", mock_am), \
         patch("server._server_meta", mock_meta), \
         patch("server.load_active_jira_profile", return_value=_make_jira_profile()):

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def run_query():
                return await client.post(
                    "/query",
                    json={"query": "slow query", "request_id": request_id},
                    timeout=5.0,
                )

            async def send_cancel():
                # Wait until the query handler has registered its task
                await cancel_event.wait()
                return await client.post(
                    "/event",
                    json={"event": "cancel", "request_id": request_id},
                )

            query_task = asyncio.create_task(run_query())
            cancel_task = asyncio.create_task(send_cancel())

            query_resp, cancel_resp = await asyncio.gather(query_task, cancel_task)

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["accepted"] is True

    assert query_resp.status_code == 200
    body = query_resp.json()
    assert "cancelled" in body.get("answer", "").lower()


@pytest.mark.asyncio
async def test_cancel_after_query_completes_returns_not_accepted():
    """
    Sending a cancel for a request_id that has already finished must return
    accepted=False — the task is no longer in the registry.
    """
    from server import app

    mock_am = _make_atlasmind_mock()
    mock_meta = _make_server_meta()
    request_id = "stale-cancel-uuid"

    with patch("server._atlasmind", mock_am), \
         patch("server._server_meta", mock_meta), \
         patch("server.load_active_jira_profile", return_value=_make_jira_profile()):

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/query",
                json={"query": "fast query", "request_id": request_id},
            )
            resp = await client.post(
                "/event",
                json={"event": "cancel", "request_id": request_id},
            )

    assert resp.status_code == 200
    assert resp.json()["accepted"] is False


@pytest.mark.asyncio
async def test_multiple_concurrent_queries_cancel_independently():
    """
    Two in-flight queries with different request_ids: cancelling one must not
    affect the other.
    """
    from server import app

    both_ready = asyncio.Barrier(2)
    results = {}

    async def slow_generate(query, jira_token=None, jira_url=None):
        await both_ready.wait()     # ensure both tasks are registered before either is cancelled
        await asyncio.sleep(10)
        return _make_llm_result("completed"), None

    mock_am = MagicMock()
    mock_am.generate_jql = slow_generate
    mock_am.field_resolver = None
    mock_am.standard_field_ids = []
    mock_am.llm_client.timeout = 300
    mock_meta = _make_server_meta()

    with patch("server._atlasmind", mock_am), \
         patch("server._server_meta", mock_meta), \
         patch("server.load_active_jira_profile", return_value=_make_jira_profile()):

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def run_query(req_id):
                r = await client.post(
                    "/query",
                    json={"query": "concurrent", "request_id": req_id},
                    timeout=5.0,
                )
                results[req_id] = r.json()

            async def cancel_one():
                await asyncio.sleep(0.1)    # let both queries register
                await client.post("/event", json={"event": "cancel", "request_id": "q1"})

            await asyncio.gather(
                asyncio.create_task(run_query("q1")),
                asyncio.create_task(run_query("q2")),
                asyncio.create_task(cancel_one()),
            )

    assert "cancelled" in results["q1"].get("answer", "").lower()
    assert results["q2"].get("answer", "").lower() == "completed"
