"""
tests/test_jira_cloud_request_body.py

Validates the exact JSON body _search_cloud sends on every request, the expand
field type fix (string not list), HTTP error handling in _fetch_page_cloud, and
server path isolation. Cloud path only - server logic is untouched.
"""

import json
import logging

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cloud.tls import tls
from jira.jira_search import JiraPage, JiraSearchClient, JiraSearchRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cloud_request(**kwargs) -> JiraSearchRequest:
    defaults = dict(
        jql="project = ABCD ORDER BY created DESC",
        fields="summary,status,assignee",
        max_results=10,
        base_url="https://sample-domain.atlassian.net",
        jira_type="cloud",
        auth=("user@example.com", "token123"),
    )
    defaults.update(kwargs)
    return JiraSearchRequest(**defaults)


def _page(issues=None, total=5, next_token=None) -> JiraPage:
    return JiraPage(
        issues=issues or [{"key": "ABCD-1", "fields": {}}],
        total=total,
        start_at=0,
        max_results=10,
        next_page_token=next_token,
    )


def _http_mock(status_code: int, body: dict | str) -> MagicMock:
    """Return a mock tls.httpx_client context manager yielding a real httpx.Response."""
    content = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
    dummy_req = httpx.Request("POST", "https://sample-domain.atlassian.net/rest/api/3/search/jql")
    response = httpx.Response(status_code, content=content, request=dummy_req)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


def _captured_body(mock, call_index=0) -> dict:
    """Extract the body dict passed to the mock _fetch_page_cloud call."""
    return mock.call_args_list[call_index][0][1]


# ---------------------------------------------------------------------------
# Section 1: expand field - type and value
# ---------------------------------------------------------------------------

class TestExpandField:
    """expand must be a comma-separated string, never a list."""

    @pytest.mark.asyncio
    async def test_no_cmdb_fields_no_expand_key(self):
        """expand must be absent when cmdb_field_ids is empty."""
        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(_cloud_request())
        assert "expand" not in _captured_body(mock)

    @pytest.mark.asyncio
    async def test_single_cmdb_field_is_string(self):
        """expand must be a str when one CMDB field ID is given."""
        client = JiraSearchClient()
        req = _cloud_request(cmdb_field_ids={"customfield_10200"})
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)

        expand = _captured_body(mock)["expand"]
        assert isinstance(expand, str), f"expand must be str, got {type(expand)}: {expand!r}"
        assert expand == "customfield_10200.cmdb.label"

    @pytest.mark.asyncio
    async def test_multiple_cmdb_fields_sorted_and_joined(self):
        """Multiple CMDB field IDs must be sorted alphabetically and joined with a comma."""
        client = JiraSearchClient()
        req = _cloud_request(cmdb_field_ids={"customfield_10300", "customfield_10100", "customfield_10200"})
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)

        expand = _captured_body(mock)["expand"]
        assert isinstance(expand, str)
        assert expand == (
            "customfield_10100.cmdb.label,"
            "customfield_10200.cmdb.label,"
            "customfield_10300.cmdb.label"
        )

    @pytest.mark.asyncio
    async def test_expand_not_a_list(self):
        """Regression: expand must never be a list (caused 400 on Jira Cloud v3 POST)."""
        client = JiraSearchClient()
        req = _cloud_request(cmdb_field_ids={"customfield_10200", "customfield_10300"})
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)

        expand = _captured_body(mock).get("expand")
        assert not isinstance(expand, list), "expand must not be a list"

    @pytest.mark.asyncio
    async def test_expand_serialises_as_json_string(self):
        """The body containing expand must round-trip through JSON with expand as a string."""
        client = JiraSearchClient()
        captured: list[dict] = []

        async def capture(url, body, auth, auth_headers):
            captured.append(body)
            return _page()

        req = _cloud_request(cmdb_field_ids={"customfield_10200"})
        with patch.object(client, "_fetch_page_cloud", new=capture):
            await client._search_cloud(req)

        parsed = json.loads(json.dumps(captured[0]))
        assert isinstance(parsed["expand"], str)


# ---------------------------------------------------------------------------
# Section 2: full request body shape
# ---------------------------------------------------------------------------

class TestRequestBodyShape:
    """Verify every key in the request body is correct for each page."""

    @pytest.mark.asyncio
    async def test_jql_present(self):
        client = JiraSearchClient()
        req = _cloud_request(jql="project = ABCD AND status = Open")
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)
        assert _captured_body(mock)["jql"] == "project = ABCD AND status = Open"

    @pytest.mark.asyncio
    async def test_fields_parsed_as_list(self):
        """Fields string must be split and stripped into a list."""
        client = JiraSearchClient()
        req = _cloud_request(fields="summary, status , assignee")
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)
        assert _captured_body(mock)["fields"] == ["summary", "status", "assignee"]

    @pytest.mark.asyncio
    async def test_max_results_in_body(self):
        client = JiraSearchClient()
        req = _cloud_request(max_results=50)
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page(total=50))) as mock:
            await client._search_cloud(req)
        assert _captured_body(mock)["maxResults"] == 50

    @pytest.mark.asyncio
    async def test_first_page_no_next_page_token(self):
        """nextPageToken must be absent on the first page."""
        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(_cloud_request())
        assert "nextPageToken" not in _captured_body(mock)

    @pytest.mark.asyncio
    async def test_second_page_carries_next_page_token(self):
        """The cursor token from page 1 must appear in the page 2 request body."""
        client = JiraSearchClient()
        issues_p1 = [{"key": f"ABCD-{i}", "fields": {}} for i in range(5)]
        issues_p2 = [{"key": f"ABCD-{i}", "fields": {}} for i in range(5, 10)]
        page1 = JiraPage(issues=issues_p1, total=10, start_at=0, max_results=5, next_page_token="cursor-abc")
        page2 = JiraPage(issues=issues_p2, total=10, start_at=0, max_results=5, next_page_token=None)

        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(side_effect=[page1, page2])) as mock:
            await client._search_cloud(_cloud_request(max_results=10))

        assert mock.call_count == 2
        assert "nextPageToken" not in _captured_body(mock, call_index=0)
        assert _captured_body(mock, call_index=1)["nextPageToken"] == "cursor-abc"

    @pytest.mark.asyncio
    async def test_cmdb_expand_present_on_every_page(self):
        """expand must be sent on every page, not just the first."""
        client = JiraSearchClient()
        issues_p1 = [{"key": f"ABCD-{i}", "fields": {}} for i in range(5)]
        issues_p2 = [{"key": f"ABCD-{i}", "fields": {}} for i in range(5, 10)]
        page1 = JiraPage(issues=issues_p1, total=10, start_at=0, max_results=5, next_page_token="cursor-abc")
        page2 = JiraPage(issues=issues_p2, total=10, start_at=0, max_results=5, next_page_token=None)

        req = _cloud_request(max_results=10, cmdb_field_ids={"customfield_10200"})
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(side_effect=[page1, page2])) as mock:
            await client._search_cloud(req)

        for i in range(2):
            body = _captured_body(mock, call_index=i)
            assert isinstance(body.get("expand"), str)

    @pytest.mark.asyncio
    async def test_empty_fields_string_becomes_empty_list(self):
        """An empty or whitespace-only fields string is stripped to an empty list."""
        client = JiraSearchClient()
        for fields_value in ("", "  ", ",,,"):
            req = _cloud_request(fields=fields_value)
            with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
                await client._search_cloud(req)
            assert _captured_body(mock)["fields"] == [], \
                f"fields={fields_value!r} must yield an empty list, got {_captured_body(mock)['fields']!r}"


# ---------------------------------------------------------------------------
# Section 3: HTTP error handling in _fetch_page_cloud
# ---------------------------------------------------------------------------

class TestFetchPageCloudErrors:
    """_fetch_page_cloud must raise ValueError with the Jira error text on HTTP errors."""

    @pytest.mark.asyncio
    async def test_400_raises_with_jira_error_message(self):
        mock_ctx = _http_mock(400, {"errorMessages": ["Field 'bogus' does not exist"], "errors": {}})
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError, match="Field 'bogus' does not exist"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    ("user@example.com", "token"),
                    {},
                )

    @pytest.mark.asyncio
    async def test_400_invalid_expand_raises_value_error(self):
        """Simulates Jira rejecting an invalid expand value."""
        mock_ctx = _http_mock(400, {"errorMessages": ["Invalid expand value: customfield_10200.cmdb.label"], "errors": {}})
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError, match="Invalid expand value"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10,
                     "expand": "customfield_10200.cmdb.label"},
                    ("user@example.com", "token"),
                    {},
                )

    @pytest.mark.asyncio
    async def test_401_raises_value_error(self):
        mock_ctx = _http_mock(401, {"errorMessages": ["You do not have permission to access this resource"], "errors": {}})
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    None,
                    {},
                )

    @pytest.mark.asyncio
    async def test_connection_error_raises_with_prefix(self):
        """Network failure must raise ValueError starting with 'Jira connection failed'."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError, match="Jira connection failed"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    None,
                    {},
                )

    @pytest.mark.asyncio
    async def test_timeout_raises_with_prefix(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError, match="Jira connection failed"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    None,
                    {},
                )

    @pytest.mark.asyncio
    async def test_200_returns_jira_page(self):
        """Successful 200 must return a JiraPage with parsed issues and total."""
        payload = {
            "issues": [{"key": "ABCD-1", "fields": {"summary": "Test"}},
                       {"key": "ABCD-2", "fields": {"summary": "Test 2"}}],
            "total": 2,
            "maxResults": 10,
            "nextPageToken": None,
        }
        mock_ctx = _http_mock(200, payload)
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            page = await client._fetch_page_cloud(
                "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                ("user@example.com", "token"),
                {},
            )

        assert len(page.issues) == 2
        assert page.total == 2
        assert page.next_page_token is None

    @pytest.mark.asyncio
    async def test_200_with_next_page_token(self):
        """next_page_token must be populated from the response when present."""
        payload = {
            "issues": [{"key": "ABCD-1", "fields": {}}],
            "total": 0,
            "maxResults": 10,
            "nextPageToken": "cursor-xyz",
        }
        mock_ctx = _http_mock(200, payload)
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            page = await client._fetch_page_cloud(
                "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                ("user@example.com", "token"),
                {},
            )

        assert page.next_page_token == "cursor-xyz"

    @pytest.mark.asyncio
    async def test_200_non_json_body_raises_json_decode_error(self):
        """A 200 response with a non-JSON (plain-text) body must raise JSONDecodeError.

        Some Atlassian load-balancer error pages return HTTP 200 with text/html
        content. httpx.Response.json() raises json.JSONDecodeError in that case,
        which is currently unhandled. This test documents the current behavior
        (crashes) so a future fix can assert a graceful error instead.
        """
        dummy_req = httpx.Request("POST", "https://sample-domain.atlassian.net/rest/api/3/search/jql")
        response = httpx.Response(
            200,
            content=b"Internal Server Error",
            request=dummy_req,
            headers={"content-type": "text/plain"},
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(json.JSONDecodeError):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    ("user@example.com", "token"),
                    {},
                )

    @pytest.mark.asyncio
    async def test_400_non_json_error_body_returns_str_exc(self):
        """A 400 with a plain-text (non-JSON) error body must still surface an error message.

        _parse_jira_error tries response.json(); if that raises JSONDecodeError the
        bare except Exception swallows it silently and falls through to str(exc).
        The ValueError message should still contain something useful (not blank).
        """
        dummy_req = httpx.Request("POST", "https://sample-domain.atlassian.net/rest/api/3/search/jql")
        response = httpx.Response(
            400,
            content=b"Proxy Error - upstream connection reset",
            request=dummy_req,
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with pytest.raises(ValueError) as exc_info:
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    ("user@example.com", "token"),
                    {},
                )

        # Must not be a blank "Jira rejected the JQL: "
        assert exc_info.value.args[0], "Error message must not be blank when Jira returns plain-text 400"
        assert "Jira rejected" in exc_info.value.args[0]


# ---------------------------------------------------------------------------
# Section 4: diagnostic logging
# ---------------------------------------------------------------------------

class TestDiagnosticLogging:
    """Verify the diagnostic log lines added for Cloud 400 debugging."""

    @pytest.mark.asyncio
    async def test_error_log_on_400_contains_status_and_body(self, caplog):
        """ERROR log must include the status code and the request body on 400."""
        mock_ctx = _http_mock(400, {"errorMessages": ["Invalid expand"], "errors": {}})
        body = {
            "jql": "project = ABCD",
            "fields": ["summary"],
            "maxResults": 10,
            "expand": "customfield_10200.cmdb.label",
        }
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with caplog.at_level(logging.ERROR, logger="jira.jira_search"):
                with pytest.raises(ValueError):
                    await client._fetch_page_cloud(
                        "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                        body,
                        ("user@example.com", "token"),
                        {},
                    )

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages, "Expected at least one ERROR log entry"
        combined = " ".join(error_messages)
        assert "400" in combined
        assert "expand" in combined

    @pytest.mark.asyncio
    async def test_debug_log_before_request(self, caplog):
        """DEBUG log must fire before every request with the request body."""
        payload = {"issues": [{"key": "ABCD-1", "fields": {}}], "total": 1, "maxResults": 10}
        mock_ctx = _http_mock(200, payload)
        body = {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10}
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with caplog.at_level(logging.DEBUG, logger="jira.jira_search"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    body,
                    ("user@example.com", "token"),
                    {},
                )

        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("request body" in m.lower() for m in debug_messages), \
            "Expected DEBUG log containing 'request body'"

    @pytest.mark.asyncio
    async def test_no_error_log_on_success(self, caplog):
        """No ERROR log must be emitted on a successful 200 response."""
        payload = {"issues": [], "total": 0, "maxResults": 10}
        mock_ctx = _http_mock(200, payload)
        client = JiraSearchClient()
        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            with caplog.at_level(logging.ERROR, logger="jira.jira_search"):
                await client._fetch_page_cloud(
                    "https://sample-domain.atlassian.net/rest/api/3/search/jql",
                    {"jql": "project = ABCD", "fields": ["summary"], "maxResults": 10},
                    ("user@example.com", "token"),
                    {},
                )

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert not error_records, f"Unexpected ERROR logs: {[r.message for r in error_records]}"


# ---------------------------------------------------------------------------
# Section 5: server path isolation
# ---------------------------------------------------------------------------

class TestServerPathIsolation:
    """Server search must never invoke cloud logic or include expand in its requests."""

    @pytest.mark.asyncio
    async def test_server_search_never_calls_fetch_page_cloud(self):
        """_search_server must never call _fetch_page_cloud."""
        client = JiraSearchClient()
        cloud_mock = AsyncMock()
        server_page = JiraPage(issues=[{"key": "ABCD-1", "fields": {}}], total=1, start_at=0, max_results=10)

        with patch.object(client, "_fetch_page_cloud", new=cloud_mock):
            with patch.object(client, "_fetch_page_server", new=AsyncMock(return_value=server_page)):
                req = JiraSearchRequest(
                    jql="project = ABCD",
                    fields="summary",
                    max_results=10,
                    base_url="https://sample-jira.example.com",
                    jira_type="server",
                    auth=("user", "token"),
                )
                await client._search_server(req)

        cloud_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cloud_search_never_calls_fetch_page_server(self):
        """_search_cloud must never call _fetch_page_server."""
        client = JiraSearchClient()
        server_mock = AsyncMock()

        with patch.object(client, "_fetch_page_server", new=server_mock):
            with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())):
                await client._search_cloud(_cloud_request())

        server_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_server_fetch_uses_get_not_post(self):
        """_fetch_page_server must send a GET request, not a POST."""
        payload = {"issues": [], "total": 0, "startAt": 0, "maxResults": 10}
        dummy_req = httpx.Request("GET", "https://sample-jira.example.com/rest/api/2/search")
        response = httpx.Response(200, content=json.dumps(payload).encode(), request=dummy_req)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=response)
        mock_client.post = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            client = JiraSearchClient()
            await client._fetch_page_server(
                url="https://sample-jira.example.com/rest/api/2/search",
                jql="project = ABCD",
                fields="summary",
                start_at=0,
                page_size=10,
                auth=("user", "token"),
                auth_headers={},
            )

        mock_client.get.assert_called_once()
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_server_request_has_no_expand_param(self):
        """Server GET request must never include an expand parameter."""
        payload = {"issues": [], "total": 0, "startAt": 0, "maxResults": 10}
        dummy_req = httpx.Request("GET", "https://sample-jira.example.com/rest/api/2/search")
        response = httpx.Response(200, content=json.dumps(payload).encode(), request=dummy_req)
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(tls, "httpx_client", return_value=mock_ctx):
            client = JiraSearchClient()
            await client._fetch_page_server(
                url="https://sample-jira.example.com/rest/api/2/search",
                jql="project = ABCD",
                fields="summary",
                start_at=0,
                page_size=10,
                auth=("user", "token"),
                auth_headers={},
            )

        call_kwargs = mock_client.get.call_args
        params = call_kwargs.kwargs.get("params", {})
        assert "expand" not in params

    @pytest.mark.asyncio
    async def test_jira_search_request_routes_cloud_to_search_cloud(self):
        """search() must route jira_type='cloud' to _search_cloud."""
        client = JiraSearchClient()
        cloud_mock = AsyncMock(return_value=MagicMock())
        server_mock = AsyncMock(return_value=MagicMock())

        req = _cloud_request()
        with patch.object(client, "_search_cloud", new=cloud_mock):
            with patch.object(client, "_search_server", new=server_mock):
                await client.search(req)

        cloud_mock.assert_called_once_with(req)
        server_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Section 6: edge cases — empty fields, malformed bodies, routing, CMDB IDs
# ---------------------------------------------------------------------------

class TestEmptyFieldsAndBodyEdgeCases:
    """Edge cases for empty inputs and malformed responses."""

    @pytest.mark.asyncio
    async def test_empty_fields_accepted_by_cloud_endpoint(self):
        """Confirm the cloud endpoint accepts fields=[] without crashing the test mock.

        The production code sends {"fields": []} when request.fields is empty/blank.
        This test documents the current behavior — Jira Cloud may silently ignore
        the fields key or return issues with no field data. A follow-up guard can
        be added to _search_cloud to default to ["summary"] when fields_list is empty.
        """
        client = JiraSearchClient()
        req = _cloud_request(fields="")
        captured: list[dict] = []

        async def capture(url, body, auth, auth_headers):
            captured.append(body)
            return _page()

        with patch.object(client, "_fetch_page_cloud", new=capture):
            # Should not raise — just document what gets sent
            result = await client._search_cloud(req)

        assert captured[0]["fields"] == []
        assert result.fetched == 1  # mock _page() returns one issue regardless of fields


class TestJiraTypeRouting:
    """search() routing behavior for various jira_type values."""

    @pytest.mark.asyncio
    async def test_unknown_jira_type_falls_through_to_server_path(self):
        """An unrecognized jira_type value silently uses the server path (current behavior).

        jira_type='cloudy' is not validated and falls to the else branch, calling
        _search_server. This is a defensive coding risk: wrong credentials could be
        sent to the wrong endpoint. A future field validator should reject unknown types.
        """
        client = JiraSearchClient()
        cloud_mock = AsyncMock()
        server_mock = AsyncMock(return_value=MagicMock())

        req = _cloud_request(jira_type="cloudy")
        with patch.object(client, "_search_cloud", new=cloud_mock):
            with patch.object(client, "_search_server", new=server_mock):
                await client.search(req)

        server_mock.assert_called_once()
        cloud_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_jira_type_falls_through_to_server_path(self):
        """jira_type='' also falls to server path by default."""
        client = JiraSearchClient()
        cloud_mock = AsyncMock()
        server_mock = AsyncMock(return_value=MagicMock())

        req = _cloud_request(jira_type="")
        with patch.object(client, "_search_cloud", new=cloud_mock):
            with patch.object(client, "_search_server", new=server_mock):
                await client.search(req)

        server_mock.assert_called_once()
        cloud_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_server_jira_type_routes_to_search_server(self):
        """jira_type='server' must always route to _search_server."""
        client = JiraSearchClient()
        cloud_mock = AsyncMock()
        server_page = JiraPage(issues=[{"key": "ABCD-1", "fields": {}}], total=1, start_at=0, max_results=10)

        req = JiraSearchRequest(
            jql="project = ABCD",
            fields="summary",
            max_results=10,
            base_url="https://sample-jira.example.com",
            jira_type="server",
            auth=("user", "token"),
        )
        with patch.object(client, "_search_cloud", new=cloud_mock):
            with patch.object(client, "_search_server", new=AsyncMock(return_value=server_page)):
                await client.search(req)

        cloud_mock.assert_not_called()


class TestCMDBFieldIDWithSpecialCharacters:
    """CMDB field IDs with special characters in the expand string."""

    @pytest.mark.asyncio
    async def test_cmdb_field_id_containing_comma_produces_ambiguous_expand(self):
        """A CMDB field ID containing a comma produces an ambiguous expand string.

        Jira's expand parameter uses comma as a delimiter, so customfield IDs that
        contain commas (e.g. 'customfield_10200,evil') cannot be reliably parsed.
        This test documents the current behavior — no validation is performed.
        A defensive fix would reject such IDs or use a different serialization.
        """
        client = JiraSearchClient()
        req = _cloud_request(cmdb_field_ids={"customfield_10200,evil"})
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=_page())) as mock:
            await client._search_cloud(req)

        expand = _captured_body(mock)["expand"]
        # The comma in the field ID creates ambiguity: "customfield_10200,evil.cmdb.label"
        # could be interpreted as two separate fields or one field with a comma in its name.
        assert "customfield_10200,evil.cmdb.label" in expand
        assert "," in expand  # confirms the problematic comma is present


class TestParseJiraErrorFallback:
    """_parse_jira_error fallback paths when response body is not a valid JSON error."""

    def test_parse_jira_error_returns_empty_str_when_both_json_and_str_exc_are_empty(self):
        """If the error body has no errorMessages/errors AND str(exc) is empty, returns ''.

        This produces a misleading "Jira rejected the JQL: " with no message.
        The function should return a non-empty string in all code paths.
        """
        from jira.jira_search import _parse_jira_error
        from unittest.mock import MagicMock

        # Craft an HTTPStatusError whose str() is empty
        dummy_req = httpx.Request("POST", "https://example.com")
        response = httpx.Response(400, content=b"{}", request=dummy_req)
        # Mock str(exc) to return empty string by using a class with __str__ returning ""
        class EmptyStrError(httpx.HTTPStatusError):
            def __str__(self):
                return ""

        exc = EmptyStrError("", request=dummy_req, response=response)
        result = _parse_jira_error(exc)

        # Current behavior: returns "" — this is the bug we document
        assert result == "", \
            "Expected empty string (documents the bug); fix _parse_jira_error to return exc.response.status_code or a default message"

    def test_parse_jira_error_returns_redirect_message_for_3xx(self):
        """A 3xx redirect should return a helpful redirect message, not a blank string."""
        from jira.jira_search import _parse_jira_error

        dummy_req = httpx.Request("POST", "https://example.com")
        response = httpx.Response(
            302,
            content=b"",
            request=dummy_req,
            headers={"location": "https://wrong-endpoint.example.com"},
        )
        exc = httpx.HTTPStatusError("", request=dummy_req, response=response)
        result = _parse_jira_error(exc)

        assert "redirect" in result.lower()
        assert "wrong-endpoint" in result
