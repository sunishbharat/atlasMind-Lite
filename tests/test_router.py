"""
tests/test_router.py — robustness tests for QueryRouter.

Covers:
  - Correct JQL routing for Jira data queries
  - Correct general routing for non-Jira queries
  - Regression cases that previously misrouted (business "issues", comparisons)
  - RouteResult properties
  - Response parsing edge cases (case, whitespace, multi-line)
  - Prompt template substitution
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.router import QueryRouter, RouteResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_PROMPT_TEMPLATE = "User query: {query}\n\nResponse:"


@pytest.fixture
def mock_llm():
    """Async LLM client stub — set mock_llm.generate_jql.return_value per test."""
    client = MagicMock()
    client.generate_jql = AsyncMock()
    return client


@pytest.fixture
def router(mock_llm, tmp_path):
    """QueryRouter wired to mock_llm and a minimal in-memory prompt file."""
    prompt_file = tmp_path / "router_prompt.md"
    prompt_file.write_text(MINIMAL_PROMPT_TEMPLATE, encoding="utf-8")
    return QueryRouter(llm_client=mock_llm, prompt_file=prompt_file)


@pytest.fixture
def real_prompt_router(mock_llm):
    """QueryRouter loaded from the real config/router_prompt.md."""
    prompt_file = Path(__file__).parent.parent / "config" / "router_prompt.md"
    return QueryRouter(llm_client=mock_llm, prompt_file=prompt_file)


# ---------------------------------------------------------------------------
# RouteResult dataclass
# ---------------------------------------------------------------------------

def test_route_result_is_jql_true():
    r = RouteResult(type="jql")
    assert r.is_jql is True


def test_route_result_is_jql_false():
    r = RouteResult(type="general", answer="Some answer.")
    assert r.is_jql is False


def test_route_result_answer_default_empty():
    r = RouteResult(type="jql")
    assert r.answer == ""


# ---------------------------------------------------------------------------
# JQL routing — queries that SHOULD trigger the JQL pipeline
# ---------------------------------------------------------------------------

JQL_QUERIES = [
    "list open bugs in project X",
    "show me tickets assigned to John",
    "what issues are unresolved in the backlog?",
    "find all epics in sprint 42",
    "show stories with priority Critical that are In Progress",
    "list tasks created this week assigned to me",
    "show me all bugs with no assignee",
    "get all tickets in the INFRA project",
    "which issues were resolved last month?",
    "show open tickets with label 'payments'",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", JQL_QUERIES)
async def test_jql_queries_route_to_jql(router, mock_llm, query):
    mock_llm.generate_jql.return_value = "JQL"
    result = await router.route(query)
    assert result.is_jql, f"Expected JQL route for: {query!r}"
    assert result.type == "jql"


# ---------------------------------------------------------------------------
# General routing — queries that MUST NOT trigger the JQL pipeline
# ---------------------------------------------------------------------------

GENERAL_QUERIES = [
    "what do you think about the issues Atlassian has in their business pipeline?",
    "compare Jira to GitHub Issues and Linear for a 30-person engineering team",
    "what is JQL?",
    "explain the difference between epics and stories",
    "list the 7 wonders of the world",
    "narrate a poem on cricket",
    "what is agile methodology?",
    "how do sprints work in scrum?",
    "what problems does our team face in general?",
    "tell me about Atlassian's product roadmap",
    "what are best practices for backlog grooming?",
    "hello",
    "what is 2 + 2?",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", GENERAL_QUERIES)
async def test_general_queries_route_to_general(router, mock_llm, query):
    answer = f"This is a helpful answer for: {query}"
    mock_llm.generate_jql.return_value = answer
    result = await router.route(query)
    assert not result.is_jql, f"Expected general route for: {query!r}"
    assert result.type == "general"
    assert result.answer == answer


# ---------------------------------------------------------------------------
# Regression: previously misrouted queries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_regression_atlassian_business_issues(router, mock_llm):
    """'issues' used in a business context must not route to JQL."""
    mock_llm.generate_jql.return_value = "Atlassian faces several business challenges..."
    result = await router.route(
        "let me know what you think about the issues that atlassian have in "
        "their pipeline related to their business"
    )
    assert not result.is_jql


@pytest.mark.asyncio
async def test_regression_tool_comparison(router, mock_llm):
    """Tool comparison queries mentioning Jira must not route to JQL."""
    mock_llm.generate_jql.return_value = "Jira is a mature tool while Linear..."
    result = await router.route(
        "Compare JIRA to GitHub Issues and Linear for a 30-person engineering team."
    )
    assert not result.is_jql


@pytest.mark.asyncio
async def test_regression_general_answer_content_preserved(router, mock_llm):
    """The full LLM response must be returned as the answer for general queries."""
    expected = "Here are the 7 wonders: 1. Great Wall..."
    mock_llm.generate_jql.return_value = expected
    result = await router.route("list the 7 wonders of the world")
    assert result.answer == expected


# ---------------------------------------------------------------------------
# Response parsing — JQL signal detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("llm_response", [
    "JQL",
    "jql",
    "Jql",
    "JQL ",
    "  JQL",   # leading whitespace stripped by router before regex
    "JQL\n",
])
async def test_jql_signal_case_and_whitespace(router, mock_llm, llm_response):
    mock_llm.generate_jql.return_value = llm_response
    result = await router.route("show open bugs")
    assert result.is_jql, f"Expected JQL for response {llm_response!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize("llm_response", [
    "answer as a general question",  # the regression: model echoing a label
    "JQLS are useful",               # 'jql' not followed by word boundary
    "This is JQL: ...",              # 'jql' not at start of response
    "",                              # empty response → general
    "Sorry, I don't know.",
])
async def test_non_jql_signals_route_to_general(router, mock_llm, llm_response):
    mock_llm.generate_jql.return_value = llm_response
    result = await router.route("some query")
    assert not result.is_jql, f"Expected general for response {llm_response!r}"


# ---------------------------------------------------------------------------
# Prompt template substitution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prompt_template_substitutes_query(mock_llm, tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Q: {query}\nA:", encoding="utf-8")
    mock_llm.generate_jql.return_value = "Some answer"
    router = QueryRouter(llm_client=mock_llm, prompt_file=prompt_file)

    await router.route("show open bugs")

    call_args = mock_llm.generate_jql.call_args[0][0]
    assert "show open bugs" in call_args
    assert "{query}" not in call_args



# ---------------------------------------------------------------------------
# Real prompt file smoke test — validates the actual router_prompt.md loads
# ---------------------------------------------------------------------------

def test_real_prompt_file_loads_without_error(real_prompt_router):
    assert real_prompt_router._prompt_template
    assert "{query}" in real_prompt_router._prompt_template


@pytest.mark.asyncio
async def test_real_prompt_substitutes_query(real_prompt_router, mock_llm):
    mock_llm.generate_jql.return_value = "JQL"
    await real_prompt_router.route("list open bugs")
    sent_prompt = mock_llm.generate_jql.call_args[0][0]
    assert "list open bugs" in sent_prompt
    assert "{query}" not in sent_prompt
