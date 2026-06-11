"""
tests/test_jira_cloud_search.py

Covers _search_cloud pagination and total-field behaviour for Jira Cloud v3.

The Cloud v3 cursor API often returns total=0 in the payload even when issues
are present (cursor-based responses may omit total entirely).  The client must:
  - not stop paginating just because total=0 while nextPageToken exists
  - report total = len(fetched issues) when the API returns total=0
"""

import pytest
from unittest.mock import AsyncMock, patch

from jira.jira_search import JiraPage, JiraSearchClient, JiraSearchRequest


def _make_request(max_results: int = 2000) -> JiraSearchRequest:
    return JiraSearchRequest(
        jql="assignee = currentUser() ORDER BY created DESC",
        fields="summary,status,assignee",
        max_results=max_results,
        base_url="https://sample-domain.atlassian.net",
        jira_type="cloud",
        auth=("user@example.com", "token123"),
    )


def _fake_issue(key: str) -> dict:
    return {"key": key, "fields": {"summary": f"Issue {key}"}}


class TestCloudSearchTotalZero:
    """total=0 returned by API alongside real issues (cursor pagination quirk)."""

    @pytest.mark.asyncio
    async def test_single_page_total_zero_uses_fetched_count(self):
        issues = [_fake_issue(f"ABCD-{i}") for i in range(10)]
        page = JiraPage(issues=issues, total=0, start_at=0, max_results=10, next_page_token=None)

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=page)):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 10
        assert result.total == 10  # must equal fetched, not 0

    @pytest.mark.asyncio
    async def test_multi_page_total_zero_follows_cursor(self):
        """Pagination must continue across pages when total=0 but nextPageToken exists."""
        page1_issues = [_fake_issue(f"ABCD-{i}") for i in range(100)]
        page2_issues = [_fake_issue(f"ABCD-{i}") for i in range(100, 140)]

        page1 = JiraPage(issues=page1_issues, total=0, start_at=0, max_results=100, next_page_token="cursor-abc")
        page2 = JiraPage(issues=page2_issues, total=0, start_at=0, max_results=100, next_page_token=None)

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(side_effect=[page1, page2])):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 140
        assert result.total == 140

    @pytest.mark.asyncio
    async def test_multi_page_second_page_has_total(self):
        """If total appears on a later page, it supersedes the fallback."""
        page1_issues = [_fake_issue(f"ABCD-{i}") for i in range(100)]
        page2_issues = [_fake_issue(f"ABCD-{i}") for i in range(100, 140)]

        page1 = JiraPage(issues=page1_issues, total=0,   start_at=0, max_results=100, next_page_token="cursor-abc")
        page2 = JiraPage(issues=page2_issues, total=140, start_at=0, max_results=100, next_page_token=None)

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(side_effect=[page1, page2])):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 140
        assert result.total == 140


class TestCloudSearchTotalPopulated:
    """Normal case: API returns a meaningful total."""

    @pytest.mark.asyncio
    async def test_stops_when_fetched_reaches_total(self):
        issues = [_fake_issue(f"ABCD-{i}") for i in range(50)]
        page = JiraPage(issues=issues, total=50, start_at=0, max_results=50, next_page_token="unused-cursor")

        client = JiraSearchClient()
        fetch_mock = AsyncMock(return_value=page)
        with patch.object(client, "_fetch_page_cloud", new=fetch_mock):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 50
        assert result.total == 50
        fetch_mock.assert_called_once()  # stopped after first page

    @pytest.mark.asyncio
    async def test_stops_when_no_next_token(self):
        issues = [_fake_issue(f"ABCD-{i}") for i in range(30)]
        page = JiraPage(issues=issues, total=30, start_at=0, max_results=30, next_page_token=None)

        client = JiraSearchClient()
        fetch_mock = AsyncMock(return_value=page)
        with patch.object(client, "_fetch_page_cloud", new=fetch_mock):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 30
        fetch_mock.assert_called_once()


class TestCloudSearchMaxResultsLimit:
    """max_results must act as a hard cap regardless of total=0 + endless nextToken."""

    @pytest.mark.asyncio
    async def test_stops_at_max_results_when_total_zero(self):
        """Simulates the infinite-loop bug: total=0, nextToken always present.
        The loop must stop once max_results issues are accumulated."""
        def _page_with_token(n: int, token: str) -> JiraPage:
            return JiraPage(
                issues=[_fake_issue(f"ABCD-{n*100+i}") for i in range(10)],
                total=0,
                start_at=0,
                max_results=10,
                next_page_token=token,
            )

        # Every call returns 10 issues and a fresh nextToken — never ends without the cap.
        fetch_mock = AsyncMock(side_effect=[_page_with_token(i, f"tok-{i+1}") for i in range(20)])

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=fetch_mock):
            result = await client._search_cloud(_make_request(max_results=10))

        assert result.fetched == 10
        assert fetch_mock.call_count == 1  # exactly one page needed for max_results=10

    @pytest.mark.asyncio
    async def test_stops_at_max_results_across_multiple_pages(self):
        """max_results=25 with 10 issues per page: must stop after 3 pages (10+10+5)."""
        pages = [
            JiraPage(issues=[_fake_issue(f"ABCD-{i}") for i in range(10)],   total=0, start_at=0, max_results=10, next_page_token="tok-1"),
            JiraPage(issues=[_fake_issue(f"ABCD-{i}") for i in range(10, 20)], total=0, start_at=0, max_results=10, next_page_token="tok-2"),
            JiraPage(issues=[_fake_issue(f"ABCD-{i}") for i in range(20, 25)], total=0, start_at=0, max_results=5,  next_page_token="tok-3"),
        ]
        fetch_mock = AsyncMock(side_effect=pages)

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=fetch_mock):
            result = await client._search_cloud(_make_request(max_results=25))

        assert result.fetched == 25
        assert fetch_mock.call_count == 3


class TestCloudSearchTrueEmpty:
    """Genuine zero-result query - not a silent auth failure, but truly no issues."""

    @pytest.mark.asyncio
    async def test_zero_results_total_is_zero(self):
        page = JiraPage(issues=[], total=0, start_at=0, max_results=100, next_page_token=None)

        client = JiraSearchClient()
        with patch.object(client, "_fetch_page_cloud", new=AsyncMock(return_value=page)):
            result = await client._search_cloud(_make_request())

        assert result.fetched == 0
        assert result.total == 0
