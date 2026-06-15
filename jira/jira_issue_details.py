"""
Fetch per-issue details for POST /issue_details.

One concurrent pair of httpx calls per issue key: one for fields+changelog,
one for comments. Per-issue 404/403 is absorbed into not_found. Connection-level
errors propagate as ValueError and cause the entire batch to fail.
"""

import asyncio
import logging
import re
from typing import Any

import httpx

from cloud.tls import tls
from core.models import ChangelogEntry, Comment, IssueDetail, IssueLink

logger = logging.getLogger(__name__)

_CONCURRENCY = 10


# -- Text normalisation -----------------------------------------------

def _strip_wiki_markup(text: str) -> str:
    """Remove common Jira Server wiki markup, leaving plain text."""
    text = re.sub(r"\{code[^}]*\}.*?\{code\}", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\{noformat[^}]*\}.*?\{noformat\}", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"^h[1-6]\.\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[*#]+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^|\]]+)\|[^\]]+\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    return text.strip()


def _extract_adf_text(node: Any) -> str:
    """Recursively extract plain text from an Atlassian Document Format node."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = [_extract_adf_text(child) for child in node.get("content", [])]
    return " ".join(p for p in parts if p)


def _parse_comment_body(body: Any) -> str:
    if isinstance(body, str):
        return _strip_wiki_markup(body)
    if isinstance(body, dict):
        return _extract_adf_text(body)
    return str(body) if body is not None else ""


# -- Parsers ----------------------------------------------------------

def _parse_comments(raw: list[dict], limit: int) -> list[Comment]:
    """Return up to limit comments, newest first."""
    sorted_raw = sorted(raw, key=lambda c: c.get("updated", c.get("created", "")), reverse=True)
    result = []
    for c in sorted_raw[:limit]:
        author_obj = c.get("author", {})
        author = author_obj.get("displayName") or author_obj.get("name") or "unknown"
        result.append(Comment(
            id=str(c.get("id", "")),
            author=author,
            body=_parse_comment_body(c.get("body")),
            created=c.get("created", ""),
            updated=c.get("updated", c.get("created", "")),
        ))
    return result


def _parse_links(raw: list[dict]) -> list[IssueLink]:
    result = []
    for link in raw:
        link_type = link.get("type", {})
        type_name = link_type.get("name", "")
        outward = link.get("outwardIssue")
        inward = link.get("inwardIssue")
        if outward:
            result.append(IssueLink(
                type=link_type.get("outward", type_name),
                direction="outward",
                linked_issue_key=outward.get("key", ""),
                linked_issue_summary=outward.get("fields", {}).get("summary"),
            ))
        elif inward:
            result.append(IssueLink(
                type=link_type.get("inward", type_name),
                direction="inward",
                linked_issue_key=inward.get("key", ""),
                linked_issue_summary=inward.get("fields", {}).get("summary"),
            ))
    return result


def _parse_changelog(histories: list[dict]) -> list[ChangelogEntry]:
    """Extract status-field transitions only, in chronological order."""
    result = []
    for history in histories:
        author_obj = history.get("author", {})
        author = author_obj.get("displayName") or author_obj.get("name") or "unknown"
        timestamp = history.get("created", "")
        for item in history.get("items", []):
            if item.get("field") == "status":
                result.append(ChangelogEntry(
                    field="status",
                    from_value=item.get("fromString"),
                    to_value=item.get("toString", ""),
                    author=author,
                    timestamp=timestamp,
                ))
    return result


def _parse_flagged(fields: dict) -> bool:
    # customfield_10021 is the standard Jira "Flagged" field; non-empty list = flagged
    val = fields.get("customfield_10021")
    if isinstance(val, list):
        return len(val) > 0
    if isinstance(val, str):
        return bool(val.strip())
    return False


def _parse_fix_versions(fields: dict) -> list[str]:
    versions = fields.get("fixVersions") or []
    return [v.get("name", "") for v in versions if isinstance(v, dict) and v.get("name")]


# -- Single-issue fetch -----------------------------------------------

_ISSUE_FIELDS = "priority,assignee,duedate,fixVersions,issuelinks,customfield_10021,comment"


async def _fetch_single_issue(
    key: str,
    base_url: str,
    auth: tuple[str, str] | None,
    auth_headers: dict[str, str],
    jira_type: str,
    comments_limit: int,
) -> tuple[str, IssueDetail | None]:
    """
    Fetch fields+changelog and comments for one issue key concurrently.

    Returns (key, IssueDetail) on success, (key, None) when the key is not
    found or inaccessible (404/403). Raises ValueError on connection failure.
    """
    api_v = "3" if jira_type == "cloud" else "2"
    resolved_auth = auth if auth and any(auth) else None
    base_headers = {"Accept": "application/json", **auth_headers}
    issue_url   = f"{base_url}/rest/api/{api_v}/issue/{key}"
    comment_url = f"{base_url}/rest/api/{api_v}/issue/{key}/comment"

    try:
        async with tls.httpx_client(timeout=20) as client:
            issue_result, comment_result = await asyncio.gather(
                client.get(
                    issue_url,
                    params={"fields": _ISSUE_FIELDS, "expand": "changelog"},
                    auth=resolved_auth,
                    headers=base_headers,
                ),
                client.get(
                    comment_url,
                    params={"maxResults": comments_limit, "orderBy": "-created"},
                    auth=resolved_auth,
                    headers=base_headers,
                ),
                return_exceptions=True,
            )
    except Exception as exc:
        raise ValueError(f"Jira connection failed: {exc}") from exc

    if isinstance(issue_result, Exception):
        raise ValueError(f"Jira connection failed: {issue_result}") from issue_result

    issue_resp = issue_result
    if isinstance(comment_result, Exception):
        logger.warning("Comment fetch for %s failed: %s - falling back to issue fields", key, comment_result)
        comment_resp = None
    else:
        comment_resp = comment_result

    if issue_resp.status_code in (404, 403):
        return key, None
    try:
        issue_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"Jira error fetching {key}: {exc}") from exc

    data   = issue_resp.json()
    fields = data.get("fields", {})

    assignee_obj = fields.get("assignee")
    assignee = (assignee_obj.get("displayName") or assignee_obj.get("name")) if assignee_obj else None

    priority_obj = fields.get("priority")
    priority = priority_obj.get("name") if priority_obj else None

    # Comments: primary source is the comment sub-endpoint; fall back to the comment field
    # returned by the issue endpoint (included in _ISSUE_FIELDS) when the sub-endpoint fails.
    # Fallback note: Jira embeds only its default page (~10 comments) in the issue response,
    # so comments_limit is not honoured in the fallback path.
    if comment_resp is not None and comment_resp.status_code == 200:
        raw_comments = comment_resp.json().get("comments", [])
    else:
        if comment_resp is not None:
            logger.warning("Comment fetch for %s returned HTTP %s - falling back to issue fields", key, comment_resp.status_code)
        raw_comments = fields.get("comment", {}).get("comments", []) if isinstance(fields.get("comment"), dict) else []
        logger.warning(
            "Comment fetch for %s using issue-fields fallback; got %d comment(s) "
            "(comments_limit=%d may not be honoured - Jira embeds only its default page)",
            key, len(raw_comments), comments_limit,
        )

    changelog_section = data.get("changelog", {})
    raw_histories = changelog_section.get("histories", []) if isinstance(changelog_section, dict) else []

    return key, IssueDetail(
        key=key,
        priority=priority,
        assignee=assignee,
        due_date=fields.get("duedate"),
        fix_versions=_parse_fix_versions(fields),
        flagged=_parse_flagged(fields),
        comments=_parse_comments(raw_comments, comments_limit),
        links=_parse_links(fields.get("issuelinks") or []),
        changelog=_parse_changelog(raw_histories),
    )


# -- Batch fetch ------------------------------------------------------

async def fetch_issue_details(
    issue_keys: list[str],
    base_url: str,
    auth: tuple[str, str] | None,
    auth_headers: dict[str, str],
    jira_type: str,
    comments_limit: int,
) -> tuple[list[IssueDetail], list[str]]:
    """
    Fetch details for multiple issue keys concurrently (bounded to _CONCURRENCY).

    Returns (issues, not_found). Raises ValueError if a connection-level error
    occurs (Jira unreachable), which the caller surfaces as a batch error.
    """
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _bounded(key: str):
        async with sem:
            return await _fetch_single_issue(key, base_url, auth, auth_headers, jira_type, comments_limit)

    results = await asyncio.gather(*[_bounded(k) for k in issue_keys], return_exceptions=True)

    issues: list[IssueDetail] = []
    not_found: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            raise r
        key, detail = r
        if detail is not None:
            issues.append(detail)
        else:
            not_found.append(key)
    return issues, not_found
