"""
tests/test_issue_details.py

Validates the POST /issue_details endpoint implementation across six sections:

  Section 1 - Pydantic models (IssueDetailsRequest, IssueDetailsResponse, etc.)
  Section 2 - Parser unit tests (pure Python, no network)
  Section 3 - _fetch_single_issue (mocked HTTP)
  Section 4 - fetch_issue_details batch behaviour (mocked _fetch_single_issue)
  Section 5 - POST /issue_details endpoint (TestClient)
  Section 6 - _resolve_base_url helper
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from core.jira_auth import JiraProfile
from core.models import (
    ChangelogEntry, Comment, IssueDetail, IssueDetailsRequest,
    IssueDetailsResponse, IssueLink,
)
from jira.jira_issue_details import (
    _extract_adf_text,
    _fetch_single_issue,
    _parse_changelog,
    _parse_comment_body,
    _parse_comments,
    _parse_fix_versions,
    _parse_flagged,
    _parse_links,
    _strip_wiki_markup,
    fetch_issue_details,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_issue_response(
    key: str = "ABCD-101",
    status: int = 200,
    extra_fields: dict | None = None,
    histories: list | None = None,
) -> MagicMock:
    fields: dict = {
        "priority":          {"name": "Critical"},
        "assignee":          {"displayName": "John Doe", "name": "jdoe"},
        "duedate":           "2026-07-15",
        "fixVersions":       [{"name": "v2.3"}],
        "issuelinks":        [],
        "customfield_10021": [{"value": "Impediment"}],
        "comment":           {"comments": []},
    }
    if extra_fields:
        fields.update(extra_fields)
    data = {
        "key": key,
        "fields": fields,
        "changelog": {"histories": histories or []},
    }
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=httpx.Request("GET", f"http://jira.test/issue/{key}"),
            response=resp,
        )
    return resp


def _make_comment_response(comments: list | None = None, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"comments": comments or []}
    return resp


def _make_tls_mock(issue_resp: MagicMock, comment_resp: MagicMock) -> MagicMock:
    """Mock tls whose httpx_client yields a client that routes GET by URL."""
    async def get(url, **kwargs):
        return comment_resp if url.endswith("/comment") else issue_resp

    mock_client = AsyncMock()
    mock_client.get.side_effect = get

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    mock_tls = MagicMock()
    mock_tls.httpx_client.return_value = ctx
    return mock_tls


def _sample_issue_detail(key: str = "ABCD-101") -> IssueDetail:
    return IssueDetail(
        key=key,
        priority="Critical",
        assignee="jdoe",
        due_date="2026-07-15",
        fix_versions=["v2.3"],
        flagged=True,
        comments=[Comment(
            id="1001", author="jdoe", body="Blocked",
            created="2026-06-01T10:00:00.000+0000",
            updated="2026-06-01T10:00:00.000+0000",
        )],
        links=[IssueLink(
            type="blocks", direction="outward",
            linked_issue_key="ABCD-102", linked_issue_summary="Sign-off",
        )],
        changelog=[ChangelogEntry(
            field="status", from_value="Open", to_value="Blocked",
            author="jdoe", timestamp="2026-05-25T10:00:00.000+0000",
        )],
    )


def _make_profile(jira_url: str = "http://jira.test", jira_type: str = "server") -> JiraProfile:
    return JiraProfile(name="test", jira_url=jira_url, jira_type=jira_type)


# ---------------------------------------------------------------------------
# Section 1 - Pydantic models
# ---------------------------------------------------------------------------

class TestIssueDetailsRequest:
    def test_issue_keys_required(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest()

    def test_empty_issue_keys_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest(issue_keys=[])

    def test_comments_limit_must_be_positive(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest(issue_keys=["ABCD-1"], comments_limit=0)

    def test_invalid_key_format_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest(issue_keys=["not-a-valid-key"])

    def test_empty_string_key_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest(issue_keys=[""])

    def test_lowercase_key_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IssueDetailsRequest(issue_keys=["abcd-1"])

    def test_comments_limit_optional(self):
        req = IssueDetailsRequest(issue_keys=["ABCD-1"])
        assert req.comments_limit is None

    def test_request_id_optional(self):
        req = IssueDetailsRequest(issue_keys=["ABCD-1"])
        assert req.request_id is None

    def test_all_fields_accepted(self):
        req = IssueDetailsRequest(
            issue_keys=["ABCD-1", "ABCD-2"],
            request_id="uuid-xyz",
            comments_limit=10,
        )
        assert req.issue_keys == ["ABCD-1", "ABCD-2"]
        assert req.request_id == "uuid-xyz"
        assert req.comments_limit == 10


class TestIssueDetailsResponseDefaults:
    def test_defaults_are_empty(self):
        resp = IssueDetailsResponse()
        assert resp.issues == []
        assert resp.not_found == []
        assert resp.error is None

    def test_error_can_be_set(self):
        resp = IssueDetailsResponse(error="Error: Jira down")
        assert resp.error == "Error: Jira down"
        assert resp.issues == []


# ---------------------------------------------------------------------------
# Section 2 - Parser unit tests
# ---------------------------------------------------------------------------

class TestStripWikiMarkup:
    def test_removes_code_block(self):
        result = _strip_wiki_markup("{code:java}\nSystem.out.println();\n{code}")
        assert "{code" not in result

    def test_removes_noformat_block(self):
        result = _strip_wiki_markup("{noformat}\nraw content\n{noformat}")
        assert "{noformat" not in result

    def test_removes_inline_macro(self):
        result = _strip_wiki_markup("{color:red}urgent{color}")
        assert "{color" not in result

    def test_removes_heading_marker(self):
        result = _strip_wiki_markup("h2. Summary\nsome text")
        assert "h2." not in result
        assert "Summary" in result

    def test_removes_bullet_marker(self):
        result = _strip_wiki_markup("* item one\n* item two")
        assert result.strip().startswith("item")

    def test_link_with_pipe_uses_display_text(self):
        assert _strip_wiki_markup("[click here|http://example.com]") == "click here"

    def test_plain_text_unchanged(self):
        text = "Simple comment with no markup."
        assert _strip_wiki_markup(text) == text


class TestExtractAdfText:
    def test_simple_text_node(self):
        assert _extract_adf_text({"type": "text", "text": "Hello world"}) == "Hello world"

    def test_nested_paragraph(self):
        node = {"type": "paragraph", "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ]}
        result = _extract_adf_text(node)
        assert "Hello" in result and "world" in result

    def test_non_dict_returns_empty(self):
        assert _extract_adf_text("not a dict") == ""
        assert _extract_adf_text(None) == ""
        assert _extract_adf_text(42) == ""

    def test_empty_content_returns_empty(self):
        assert _extract_adf_text({"type": "paragraph", "content": []}) == ""


class TestParseCommentBody:
    def test_string_strips_wiki_markup(self):
        result = _parse_comment_body("{code}int x = 1;{code} rest")
        assert "{code" not in result

    def test_dict_extracts_adf_text(self):
        body = {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]}
        assert _parse_comment_body(body) == "Hello"

    def test_none_returns_empty(self):
        assert _parse_comment_body(None) == ""


class TestParseComments:
    def _raw(self, cid: str, updated: str, body: str = "text") -> dict:
        return {
            "id": cid,
            "author": {"displayName": "User"},
            "body": body,
            "created": "2026-01-01T00:00:00.000+0000",
            "updated": updated,
        }

    def test_newest_first_ordering(self):
        raw = [
            self._raw("1", "2026-01-01T10:00:00.000+0000"),
            self._raw("2", "2026-01-03T10:00:00.000+0000"),
            self._raw("3", "2026-01-02T10:00:00.000+0000"),
        ]
        assert [c.id for c in _parse_comments(raw, limit=10)] == ["2", "3", "1"]

    def test_limit_enforced(self):
        raw = [self._raw(str(i), f"2026-01-0{i+1}T00:00:00.000+0000") for i in range(5)]
        assert len(_parse_comments(raw, limit=3)) == 3

    def test_display_name_preferred_over_username(self):
        raw = [{"id": "1", "author": {"displayName": "John Doe", "name": "jdoe"},
                "body": "hi", "created": "2026-01-01T00:00:00.000+0000",
                "updated": "2026-01-01T00:00:00.000+0000"}]
        assert _parse_comments(raw, limit=5)[0].author == "John Doe"

    def test_empty_list(self):
        assert _parse_comments([], limit=10) == []


class TestParseLinks:
    def test_outward_link(self):
        raw = [{"type": {"name": "Blocks", "outward": "blocks"},
                "outwardIssue": {"key": "ABCD-2", "fields": {"summary": "Sign-off"}}}]
        result = _parse_links(raw)
        assert len(result) == 1
        assert result[0].direction == "outward"
        assert result[0].linked_issue_key == "ABCD-2"
        assert result[0].linked_issue_summary == "Sign-off"

    def test_inward_link(self):
        raw = [{"type": {"name": "Blocks", "inward": "is blocked by"},
                "inwardIssue": {"key": "ABCD-5", "fields": {"summary": "Blocker"}}}]
        result = _parse_links(raw)
        assert result[0].direction == "inward"
        assert result[0].type == "is blocked by"

    def test_missing_summary_is_none(self):
        raw = [{"type": {"name": "relates to", "outward": "relates to"},
                "outwardIssue": {"key": "ABCD-9", "fields": {}}}]
        assert _parse_links(raw)[0].linked_issue_summary is None

    def test_empty_list(self):
        assert _parse_links([]) == []


class TestParseChangelog:
    def _hist(self, items: list, ts: str = "2026-05-01T00:00:00.000+0000") -> dict:
        return {"author": {"displayName": "jdoe"}, "created": ts, "items": items}

    def test_status_change_extracted(self):
        result = _parse_changelog([self._hist(
            [{"field": "status", "fromString": "Open", "toString": "In Progress"}]
        )])
        assert len(result) == 1
        assert result[0].field == "status"
        assert result[0].from_value == "Open"
        assert result[0].to_value == "In Progress"

    def test_non_status_items_excluded(self):
        result = _parse_changelog([self._hist([
            {"field": "status",   "fromString": "Open",  "toString": "Done"},
            {"field": "assignee", "fromString": "alice", "toString": "bob"},
            {"field": "priority", "fromString": "Low",   "toString": "High"},
        ])])
        assert len(result) == 1
        assert result[0].to_value == "Done"

    def test_chronological_order_preserved(self):
        result = _parse_changelog([
            self._hist([{"field": "status", "fromString": "Open",        "toString": "In Progress"}], "2026-01-01T00:00:00.000+0000"),
            self._hist([{"field": "status", "fromString": "In Progress", "toString": "Done"}],        "2026-02-01T00:00:00.000+0000"),
        ])
        assert result[0].to_value == "In Progress"
        assert result[1].to_value == "Done"

    def test_empty_histories(self):
        assert _parse_changelog([]) == []


class TestParseFlagged:
    def test_non_empty_list_is_flagged(self):
        assert _parse_flagged({"customfield_10021": [{"value": "Impediment"}]}) is True

    def test_empty_list_is_not_flagged(self):
        assert _parse_flagged({"customfield_10021": []}) is False

    def test_missing_field_is_not_flagged(self):
        assert _parse_flagged({}) is False

    def test_none_value_is_not_flagged(self):
        assert _parse_flagged({"customfield_10021": None}) is False

    def test_non_empty_string_is_flagged(self):
        assert _parse_flagged({"customfield_10021": "Impediment"}) is True


class TestParseFixVersions:
    def test_names_extracted(self):
        assert _parse_fix_versions({"fixVersions": [{"name": "v2.3"}, {"name": "v2.4"}]}) == ["v2.3", "v2.4"]

    def test_none_field_returns_empty(self):
        assert _parse_fix_versions({"fixVersions": None}) == []

    def test_missing_field_returns_empty(self):
        assert _parse_fix_versions({}) == []

    def test_entry_without_name_skipped(self):
        assert _parse_fix_versions({"fixVersions": [{"name": "v1.0"}, {"id": "99"}]}) == ["v1.0"]


# ---------------------------------------------------------------------------
# Section 3 - _fetch_single_issue (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchSingleIssue:
    @pytest.mark.asyncio
    async def test_success_returns_populated_detail(self):
        issue_resp = _make_issue_response("ABCD-101")
        comment_resp = _make_comment_response([{
            "id": "1", "author": {"displayName": "jdoe"},
            "body": "blocked",
            "created": "2026-06-01T00:00:00.000+0000",
            "updated": "2026-06-01T00:00:00.000+0000",
        }])

        with patch("jira.jira_issue_details.tls", _make_tls_mock(issue_resp, comment_resp)):
            key, detail = await _fetch_single_issue(
                "ABCD-101", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert key == "ABCD-101"
        assert detail is not None
        assert detail.key == "ABCD-101"
        assert detail.priority == "Critical"
        assert detail.assignee == "John Doe"
        assert detail.due_date == "2026-07-15"
        assert detail.fix_versions == ["v2.3"]
        assert detail.flagged is True
        assert len(detail.comments) == 1
        assert detail.comments[0].body == "blocked"

    @pytest.mark.asyncio
    async def test_404_returns_none_detail(self):
        with patch("jira.jira_issue_details.tls",
                   _make_tls_mock(_make_issue_response(status=404), _make_comment_response())):
            key, detail = await _fetch_single_issue(
                "ABCD-999", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert key == "ABCD-999"
        assert detail is None

    @pytest.mark.asyncio
    async def test_403_returns_none_detail(self):
        with patch("jira.jira_issue_details.tls",
                   _make_tls_mock(_make_issue_response(status=403), _make_comment_response())):
            _, detail = await _fetch_single_issue(
                "ABCD-403", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert detail is None

    @pytest.mark.asyncio
    async def test_connection_error_raises_value_error(self):
        async def failing_get(url, **kwargs):
            raise httpx.ConnectError("connection refused")

        mock_client = AsyncMock()
        mock_client.get.side_effect = failing_get
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_tls = MagicMock()
        mock_tls.httpx_client.return_value = ctx

        with patch("jira.jira_issue_details.tls", mock_tls):
            with pytest.raises(ValueError, match="Jira connection failed"):
                await _fetch_single_issue(
                    "ABCD-1", "http://jira.test", None, {}, "server", comments_limit=5
                )

    @pytest.mark.asyncio
    async def test_comment_connection_error_falls_back_to_issue_fields(self):
        """httpx connection error on comment endpoint falls back gracefully without losing the issue."""
        comment_in_fields = [{
            "id": "77", "author": {"displayName": "fallback"},
            "body": "from issue fields",
            "created": "2026-06-01T00:00:00.000+0000",
            "updated": "2026-06-01T00:00:00.000+0000",
        }]
        issue_resp = _make_issue_response(extra_fields={"comment": {"comments": comment_in_fields}})

        async def get(url, **kwargs):
            if url.endswith("/comment"):
                raise httpx.ConnectError("comment endpoint down")
            return issue_resp

        mock_client = AsyncMock()
        mock_client.get.side_effect = get
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_tls = MagicMock()
        mock_tls.httpx_client.return_value = ctx

        with patch("jira.jira_issue_details.tls", mock_tls):
            key, detail = await _fetch_single_issue(
                "ABCD-101", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert detail is not None
        assert key == "ABCD-101"
        assert len(detail.comments) == 1
        assert detail.comments[0].body == "from issue fields"

    @pytest.mark.asyncio
    async def test_comment_fallback_when_endpoint_fails(self):
        """Non-200 from comment sub-endpoint falls back to comment data in issue fields."""
        comment_in_fields = [{
            "id": "99", "author": {"displayName": "fallback"},
            "body": "from fields",
            "created": "2026-06-01T00:00:00.000+0000",
            "updated": "2026-06-01T00:00:00.000+0000",
        }]
        issue_resp = _make_issue_response(extra_fields={"comment": {"comments": comment_in_fields}})
        comment_resp = _make_comment_response(status=500)

        with patch("jira.jira_issue_details.tls", _make_tls_mock(issue_resp, comment_resp)):
            _, detail = await _fetch_single_issue(
                "ABCD-101", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert detail is not None
        assert len(detail.comments) == 1
        assert detail.comments[0].body == "from fields"

    @pytest.mark.asyncio
    async def test_changelog_filtered_to_status_only(self):
        histories = [{"author": {"displayName": "jdoe"}, "created": "2026-05-01T00:00:00.000+0000",
                      "items": [
                          {"field": "status",   "fromString": "Open", "toString": "Blocked"},
                          {"field": "priority", "fromString": "Low",  "toString": "High"},
                      ]}]
        issue_resp = _make_issue_response(histories=histories)

        with patch("jira.jira_issue_details.tls", _make_tls_mock(issue_resp, _make_comment_response())):
            _, detail = await _fetch_single_issue(
                "ABCD-101", "http://jira.test", None, {}, "server", comments_limit=5
            )

        assert len(detail.changelog) == 1
        assert detail.changelog[0].field == "status"

    @pytest.mark.asyncio
    async def test_cloud_jira_type_uses_api_v3(self):
        called_urls: list[str] = []

        async def capturing_get(url, **kwargs):
            called_urls.append(url)
            return _make_comment_response() if "comment" in url else _make_issue_response()

        mock_client = AsyncMock()
        mock_client.get.side_effect = capturing_get
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_tls = MagicMock()
        mock_tls.httpx_client.return_value = ctx

        with patch("jira.jira_issue_details.tls", mock_tls):
            await _fetch_single_issue(
                "ABCD-1", "http://jira.test", None, {}, "cloud", comments_limit=5
            )

        assert all("/api/3/" in url for url in called_urls)


# ---------------------------------------------------------------------------
# Section 4 - fetch_issue_details batch behaviour
# ---------------------------------------------------------------------------

class TestFetchIssueDetailsBatch:
    @pytest.mark.asyncio
    async def test_all_found_returns_all_issues(self):
        d1, d2 = _sample_issue_detail("ABCD-1"), _sample_issue_detail("ABCD-2")

        async def mock_fetch(key, *a, **kw):
            return (key, d1 if key == "ABCD-1" else d2)

        with patch("jira.jira_issue_details._fetch_single_issue", side_effect=mock_fetch):
            issues, not_found = await fetch_issue_details(
                ["ABCD-1", "ABCD-2"], "http://jira.test", None, {}, "server", 20
            )

        assert len(issues) == 2
        assert not_found == []

    @pytest.mark.asyncio
    async def test_not_found_keys_split_correctly(self):
        d1 = _sample_issue_detail("ABCD-1")

        async def mock_fetch(key, *a, **kw):
            return (key, d1 if key == "ABCD-1" else None)

        with patch("jira.jira_issue_details._fetch_single_issue", side_effect=mock_fetch):
            issues, not_found = await fetch_issue_details(
                ["ABCD-1", "ABCD-999"], "http://jira.test", None, {}, "server", 20
            )

        assert len(issues) == 1
        assert "ABCD-999" in not_found

    @pytest.mark.asyncio
    async def test_value_error_propagates_as_batch_failure(self):
        async def mock_fetch(key, *a, **kw):
            raise ValueError("Jira connection failed: timeout")

        with patch("jira.jira_issue_details._fetch_single_issue", side_effect=mock_fetch):
            with pytest.raises(ValueError, match="Jira connection failed"):
                await fetch_issue_details(
                    ["ABCD-1"], "http://jira.test", None, {}, "server", 20
                )

    @pytest.mark.asyncio
    async def test_all_not_found_returns_empty_issues(self):
        async def mock_fetch(key, *a, **kw):
            return (key, None)

        with patch("jira.jira_issue_details._fetch_single_issue", side_effect=mock_fetch):
            issues, not_found = await fetch_issue_details(
                ["ABCD-1", "ABCD-2"], "http://jira.test", None, {}, "server", 20
            )

        assert issues == []
        assert set(not_found) == {"ABCD-1", "ABCD-2"}


# ---------------------------------------------------------------------------
# Section 5 - POST /issue_details endpoint (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def issue_client():
    from server import app
    from core.models import ServerMeta

    mock_am = MagicMock()
    mock_am.field_resolver = None
    mock_am.standard_field_ids = []
    mock_am.llm_backend = "ollama"
    mock_am.llm_client.timeout = 300
    mock_meta = ServerMeta(model_name="test", llm_timeout=300)

    with patch("server.AtlasMind", return_value=mock_am), \
         patch("server._server_meta", mock_meta), \
         patch("server.load_active_jira_profile", return_value=_make_profile()):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


class TestIssueDetailsEndpoint:
    def test_happy_path_returns_200_with_issues(self, issue_client):
        detail = _sample_issue_detail("ABCD-101")
        with patch("server.fetch_issue_details", new=AsyncMock(return_value=([detail], []))):
            resp = issue_client.post("/issue_details", json={"issue_keys": ["ABCD-101"]})

        assert resp.status_code == 200
        body = resp.json()
        assert body["error"] is None
        assert len(body["issues"]) == 1
        assert body["issues"][0]["key"] == "ABCD-101"
        assert body["not_found"] == []

    def test_some_not_found_split_correctly(self, issue_client):
        detail = _sample_issue_detail("ABCD-101")
        with patch("server.fetch_issue_details", new=AsyncMock(return_value=([detail], ["ABCD-999"]))):
            resp = issue_client.post("/issue_details", json={"issue_keys": ["ABCD-101", "ABCD-999"]})

        body = resp.json()
        assert len(body["issues"]) == 1
        assert body["not_found"] == ["ABCD-999"]

    def test_batch_failure_sets_error_field(self, issue_client):
        with patch("server.fetch_issue_details",
                   new=AsyncMock(side_effect=ValueError("Jira connection failed: timeout"))):
            resp = issue_client.post("/issue_details", json={"issue_keys": ["ABCD-1"]})

        assert resp.status_code == 200
        body = resp.json()
        assert body["issues"] == []
        assert body["not_found"] == []
        assert body["error"] is not None
        assert "Jira connection failed" in body["error"]

    def test_exceeds_max_keys_returns_422(self, issue_client):
        from settings import MAX_ISSUE_DETAILS_KEYS
        keys = [f"ABCD-{i}" for i in range(MAX_ISSUE_DETAILS_KEYS + 1)]
        resp = issue_client.post("/issue_details", json={"issue_keys": keys})
        assert resp.status_code == 422
        assert "issue_keys" in resp.json().get("detail", "").lower()

    def test_invalid_key_format_returns_422(self, issue_client):
        resp = issue_client.post("/issue_details", json={"issue_keys": ["not-a-valid-key"]})
        assert resp.status_code == 422

    def test_empty_issue_keys_returns_422(self, issue_client):
        resp = issue_client.post("/issue_details", json={"issue_keys": []})
        assert resp.status_code == 422

    def test_missing_issue_keys_returns_422(self, issue_client):
        resp = issue_client.post("/issue_details", json={})
        assert resp.status_code == 422

    def test_comments_limit_capped_at_server_max(self, issue_client):
        from settings import MAX_ISSUE_DETAILS_COMMENTS
        captured: dict = {}

        async def capture_fetch(issue_keys, base_url, auth, auth_headers, jira_type, comments_limit):
            captured["comments_limit"] = comments_limit
            return ([], [])

        with patch("server.fetch_issue_details", new=capture_fetch):
            issue_client.post("/issue_details", json={
                "issue_keys": ["ABCD-1"],
                "comments_limit": MAX_ISSUE_DETAILS_COMMENTS + 100,
            })

        assert captured["comments_limit"] == MAX_ISSUE_DETAILS_COMMENTS

    def test_default_comments_limit_is_20(self, issue_client):
        captured: dict = {}

        async def capture_fetch(issue_keys, base_url, auth, auth_headers, jira_type, comments_limit):
            captured["comments_limit"] = comments_limit
            return ([], [])

        with patch("server.fetch_issue_details", new=capture_fetch):
            issue_client.post("/issue_details", json={"issue_keys": ["ABCD-1"]})

        assert captured["comments_limit"] == 20

    def test_request_id_is_optional(self, issue_client):
        with patch("server.fetch_issue_details", new=AsyncMock(return_value=([], []))):
            resp = issue_client.post("/issue_details", json={"issue_keys": ["ABCD-1"]})
        assert resp.status_code == 200

    def test_valid_x_jira_url_header_overrides_profile(self, issue_client):
        captured: dict = {}

        async def capture_fetch(issue_keys, base_url, auth, auth_headers, jira_type, comments_limit):
            captured["base_url"] = base_url
            return ([], [])

        with patch("server.fetch_issue_details", new=capture_fetch):
            issue_client.post(
                "/issue_details",
                json={"issue_keys": ["ABCD-1"]},
                headers={"X-Jira-Url": "http://other-jira.test"},
            )

        assert captured["base_url"] == "http://other-jira.test"

    def test_invalid_x_jira_url_falls_back_to_profile(self, issue_client):
        captured: dict = {}

        async def capture_fetch(issue_keys, base_url, auth, auth_headers, jira_type, comments_limit):
            captured["base_url"] = base_url
            return ([], [])

        with patch("server.fetch_issue_details", new=capture_fetch):
            issue_client.post(
                "/issue_details",
                json={"issue_keys": ["ABCD-1"]},
                headers={"X-Jira-Url": "not-a-valid-url"},
            )

        assert captured["base_url"] == "http://jira.test"

    def test_response_shape_matches_contract(self, issue_client):
        detail = _sample_issue_detail("ABCD-101")
        with patch("server.fetch_issue_details", new=AsyncMock(return_value=([detail], []))):
            resp = issue_client.post("/issue_details", json={"issue_keys": ["ABCD-101"]})

        body = resp.json()
        for key in ("issues", "not_found", "error"):
            assert key in body, f"top-level field missing: {key}"
        issue = body["issues"][0]
        for field in ("key", "priority", "assignee", "due_date", "fix_versions",
                      "flagged", "comments", "links", "changelog"):
            assert field in issue, f"IssueDetail field missing: {field}"


# ---------------------------------------------------------------------------
# Section 6 - _resolve_base_url helper
# ---------------------------------------------------------------------------

class TestResolveBaseUrl:
    def _call(self, header_url: str | None, profile_url: str = "http://jira.test") -> str:
        from server import _resolve_base_url
        return _resolve_base_url(header_url, profile_url)

    def test_valid_http_header_used(self):
        assert self._call("http://other-jira.test") == "http://other-jira.test"

    def test_valid_https_header_used(self):
        assert self._call("https://cloud.atlassian.net") == "https://cloud.atlassian.net"

    def test_trailing_slash_stripped(self):
        assert self._call("http://other-jira.test/") == "http://other-jira.test"

    def test_none_falls_back_to_profile(self):
        assert self._call(None) == "http://jira.test"

    def test_invalid_scheme_falls_back_to_profile(self):
        assert self._call("ftp://invalid.test") == "http://jira.test"

    def test_missing_netloc_falls_back_to_profile(self):
        assert self._call("not-a-valid-url") == "http://jira.test"

    def test_empty_string_falls_back_to_profile(self):
        assert self._call("") == "http://jira.test"
