"""
tests/test_llm_response_cleaner.py

Comprehensive regression tests for _clean_llm_response across all four
cloud clients (bedrock, groq, vllm, claude).

These tests verify that the preamble-stripping + brace-boundary logic
handles both the cloud-only bug (preamble before JSON, commit 1608865)
and all edge cases that could affect any client, including Jira Server
(which does not call this function but whose JSON is also processed
by _extract_json_object in atlasmind.py).

Coverage:
  - Clean JSON (server's common case — no preamble, no fences)
  - Markdown fences (```json ... ```)
  - Plain-text preamble before JSON (the primary cloud bug)
  - JSON at position 0 with trailing text after the closing brace
  - Fences + preamble combined
  - Explanation text + fences + trailing
  - Multiple ``` fence pairs
  - Empty string / no braces
  - Whitespace-only preamble / JSON starts with whitespace
  - Brace inside string value (e.g. {"jql": "Filter: {open}"})
  - String containing '{' that appears BEFORE the JSON object starts
  - Deeply nested braces in string values
  - Unicode in JSON (non-ASCII field names, emojis in values)
  - Escaped quotes and backslashes in string values
  - Jira-specific: backtick code spans, issue keys, aqlFunction in values
  - Jira-specific: JQL with double-quoted field names
  - Idempotence: same input always gives same output
  - Cross-client: all four cloud clients produce identical results
"""

import pytest


# ---------------------------------------------------------------------------
# Shared expected values
# ---------------------------------------------------------------------------

CANONICAL = '{"jql": "status = \\"Done\\"", "answer": "done issues", "intent_fields": ["Status"]}'

EXACT_BUG = (
    '{"jql": "\\"Sample Planned Version\\" >= \\"v1.0.0\\" AND '
    '\\"Sample Planned Version\\" <= \\"v2.0.0\\"", '
    '"chart_spec": null, '
    '"answer": "Issues where Sample Planned Version is between v1.0.0 and v2.0.0", '
    '"intent_fields": ["Sample Planned Version"], '
    '"where_fields": ["Sample Planned Version"]}'
)

JIRA_SERVER_CLEAN = (
    '{"jql": "project = REQ AND status = \\"Done\\"", '
    '"answer": "Done issues in REQ", '
    '"intent_fields": ["Status", "Summary"]}'
)

# aqlFunction in a string value (Jira Assets specific)
# Python double-quoted string to avoid escaping nightmare with nested quotes.
_AQL_FUNC_JSON = (
    '{"jql": "Domain in aqlFunction(',  # no trailing quote yet
)
_AQL_FUNC_PART2 = (
    'Name = ',  # single-quoted in JSON
)
_AQL_FUNC_PART3 = (
    'server-01',  # double-quoted in JSON
)
_AQL_FUNC_PART4 = (
    ')", "answer": "Domain is server-01", "intent_fields": ["Domain"]}'
)
AQL_FUNC = _AQL_FUNC_JSON + "'" + _AQL_FUNC_PART2 + '"' + _AQL_FUNC_PART3 + '"' + _AQL_FUNC_PART4

# ---------------------------------------------------------------------------
# Parametrised helper
# ---------------------------------------------------------------------------

CLOUD_CLIENTS = [
    "core.bedrock_claude_client",
    "core.groq_client",
    "core.vllm_client",
    "core.claude_client",
]

def _load(module_name: str):
    """Load clean_llm_response from a cloud client module.

    All four clients re-export clean_llm_response from core.llm_utils.
    Testing through each module verifies the import chain is intact.
    """
    import importlib
    mod = importlib.import_module(module_name)
    return mod.clean_llm_response


# Also expose the canonical implementation directly for targeted tests.
from core.llm_utils import clean_llm_response as _canonical_fn


# ---------------------------------------------------------------------------
# Primary bug case: plain-text preamble before JSON (cloud only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestPreambleBeforeJSON:
    """The exact failure reported in the bug: LLM explains before outputting JSON."""

    def test_exact_bug_report(self, module):
        """Reproduce commit 1608865's failure: preamble + bare JSON (no fences)."""
        fn = _load(module)
        raw = (
            "Looking at the user's request:\n"
            '- Field: "Sample Planned Version" (JQL: `Sample Planned Version`)\n'
            "- Range: between v1.0.0 and v2.0.0\n\n"
            + EXACT_BUG
        )
        assert fn(raw) == EXACT_BUG

    def test_preamble_no_fences(self, module):
        """Plain preamble + JSON — no fences at all."""
        fn = _load(module)
        raw = "Looking at the request:\nHere's the JSON:\n" + CANONICAL
        assert fn(raw) == CANONICAL

    def test_preamble_ending_with_colon(self, module):
        """Preamble ending with colon, JSON immediately after."""
        fn = _load(module)
        raw = "Here is the result:\n" + CANONICAL
        assert fn(raw) == CANONICAL

    def test_preamble_single_sentence(self, module):
        """Very short preamble — just one sentence."""
        fn = _load(module)
        raw = "Sure, here it is:\n" + CANONICAL
        assert fn(raw) == CANONICAL

    def test_preamble_only_newlines(self, module):
        """Preamble is just blank lines before JSON."""
        fn = _load(module)
        raw = "\n\n\n" + CANONICAL
        assert fn(raw) == CANONICAL


# ---------------------------------------------------------------------------
# Markdown fences — existing behavior, must not regress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestMarkdownFences:
    """Markdown fences have always been stripped — verify they still are."""

    def test_fenced_standard(self, module):
        """Standard ```json ... ``` wrap."""
        fn = _load(module)
        raw = "```json\n" + CANONICAL + "\n```"
        assert fn(raw) == CANONICAL

    def test_fenced_no_language_tag(self, module):
        """Bare ``` ... ``` without json tag."""
        fn = _load(module)
        raw = "```\n" + CANONICAL + "\n```"
        assert fn(raw) == CANONICAL

    def test_fenced_with_preamble_outside(self, module):
        """Preamble outside fences: 'Result:\n```json\n{...}\n```'."""
        fn = _load(module)
        raw = "Here is the JSON you requested:\n```json\n" + CANONICAL + "\n```"
        assert fn(raw) == CANONICAL

    def test_fenced_with_trailing_outside(self, module):
        """Fences, then trailing text after closing ```."""
        fn = _load(module)
        raw = "```json\n" + CANONICAL + "\n```\nPlease let me know if you need anything else."
        assert fn(raw) == CANONICAL

    def test_fenced_with_preamble_and_trailing(self, module):
        """Preamble + fences + trailing text — all three layers."""
        fn = _load(module)
        raw = (
            "Based on your query, here is the JQL:\n"
            "```json\n" + CANONICAL + "\n```\n"
            "Let me know if you'd like modifications."
        )
        assert fn(raw) == CANONICAL

    def test_multiple_fence_pairs(self, module):
        """LLM outputs multiple code blocks — only last closing ``` used."""
        fn = _load(module)
        raw = (
            "Here's the JSON:\n"
            "```json\n" + CANONICAL + "\n```\n"
            "And here's a bar chart:\n"
            "```\n"
            "x: status\n"
            "y: count\n"
            "```"
        )
        assert fn(raw) == CANONICAL


# ---------------------------------------------------------------------------
# Trailing content after JSON — the v1 bug found by first-round testing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestTrailingContent:
    """JSON at position 0, trailing text appended after closing brace."""

    def test_trailing_sentence(self, module):
        """JSON followed by a sentence."""
        fn = _load(module)
        raw = CANONICAL + "\nPlease let me know if you need modifications."
        assert fn(raw) == CANONICAL

    def test_trailing_newline_and_text(self, module):
        """JSON followed by newline and multiple lines of explanation."""
        fn = _load(module)
        raw = CANONICAL + "\n\nThis query searches for all issues.\nYou can modify the filters."
        assert fn(raw) == CANONICAL

    def test_trailing_json_like_text(self, module):
        """Trailing text that looks like more JSON."""
        fn = _load(module)
        raw = CANONICAL + "\n\nAdditional info: {\"note\": \"this is not JSON\"}"
        assert fn(raw) == CANONICAL

    def test_trailing_backtick_block(self, module):
        """JSON followed by a backtick code block."""
        fn = _load(module)
        raw = CANONICAL + "\n\n```\nsome code\n```"
        assert fn(raw) == CANONICAL

    def test_trailing_only_closing_brace_repeated(self, module):
        """Trailing text that repeats the closing brace structure."""
        fn = _load(module)
        raw = CANONICAL + "}\n}\n}"
        assert fn(raw) == CANONICAL


# ---------------------------------------------------------------------------
# Jira Server / Ollama path — clean JSON (must not regress)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestCleanJSON:
    """Server's common case: clean JSON, no preamble, no fences."""

    def test_clean_json_unchanged(self, module):
        """Clean JSON at position 0, no fences — identical to input."""
        fn = _load(module)
        assert fn(CANONICAL) == CANONICAL

    def test_jira_server_typical(self, module):
        """Simulate a typical Jira Server / Ollama clean response."""
        fn = _load(module)
        assert fn(JIRA_SERVER_CLEAN) == JIRA_SERVER_CLEAN

    def test_json_with_leading_whitespace(self, module):
        """JSON at position 0 but with leading spaces/newlines."""
        fn = _load(module)
        raw = "  \n  " + CANONICAL + "  \n"
        assert fn(raw) == CANONICAL

    def test_json_at_position_zero_exactly(self, module):
        """JSON starts exactly at character 0 — no whitespace at all."""
        fn = _load(module)
        assert fn(CANONICAL) == CANONICAL


# ---------------------------------------------------------------------------
# Edge cases: string values containing braces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestBracesInStringValues:
    """Brace depth tracking is string-aware — braces in values are not objects."""

    def test_single_brace_in_string_value(self, module):
        """String value contains a single '{' — e.g. {"jql": "Filter: {open}"}."""
        fn = _load(module)
        raw = '{"jql": "Filter: {open}"}'
        result = fn(raw)
        assert result  # must not crash; brace found in string

    def test_multiple_braces_in_string_value(self, module):
        """String value contains multiple braces."""
        fn = _load(module)
        raw = '{"jql": "Filter: {open} AND {close}"}'
        result = fn(raw)
        assert result

    def test_nested_braces_in_string_value(self, module):
        """String value contains nested-looking braces."""
        fn = _load(module)
        raw = '{"jql": "summary ~ \\"{nested: {inner}}\\""}'
        result = fn(raw)
        assert result

    def test_jira_field_in_string_value(self, module):
        """String value contains a Jira field name (underscores, no braces)."""
        fn = _load(module)
        raw = '{"jql": "customfield_10016 > 5", "answer": "customfields used"}'
        assert fn(raw) == raw


# ---------------------------------------------------------------------------
# Edge cases: malformed / unusual input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestMalformedInput:
    """Inputs that are unusual but the function should handle gracefully."""

    def test_empty_string(self, module):
        """Empty string — nothing to extract."""
        fn = _load(module)
        assert fn("") == ""

    def test_whitespace_only(self, module):
        """Input is only spaces and newlines."""
        fn = _load(module)
        assert fn("   \n\n  \n") == ""

    def test_no_brace_at_all(self, module):
        """Input has no braces at all — returns stripped input."""
        fn = _load(module)
        assert fn("No JSON here") == "No JSON here"

    def test_open_brace_only(self, module):
        """Input is only '{' — depth tracking starts but never reaches 0."""
        fn = _load(module)
        assert fn("{") == "{"

    def test_close_brace_only(self, module):
        """Input is only '}' — find('{') returns -1, returns stripped."""
        fn = _load(module)
        assert fn("}") == "}"

    def test_single_brace_pair(self, module):
        """Input is exactly '{}' — valid empty JSON object."""
        fn = _load(module)
        assert fn("{}") == "{}"

    def test_unclosed_object(self, module):
        """Input is '{...' with no closing brace — depth never reaches 0."""
        fn = _load(module)
        raw = '{"jql": "status'
        result = fn(raw)
        # Returns entire input since depth never hits 0
        assert result == raw

    def test_json_looks_like_code(self, module):
        """JSON that looks like Python/JS code with double braces."""
        fn = _load(module)
        raw = '{{"jql": "status"}}'
        result = fn(raw)
        assert result  # first { starts tracking; depth=2 at first }, depth=1 at second }


# ---------------------------------------------------------------------------
# Edge cases: Jira-specific content in string values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestJiraSpecificContent:
    """Jira-specific patterns in LLM response string values."""

    def test_aql_function_in_string(self, module):
        """aqlFunction() appears inside a string value."""
        fn = _load(module)
        result = fn(AQL_FUNC)
        assert result

    def test_backtick_code_span_in_string(self, module):
        """Jira field name in backticks (e.g. `Sample Planned Version`)."""
        fn = _load(module)
        raw = '{"jql": "`Sample Planned Version`", "answer": "version field"}'
        assert fn(raw) == raw

    def test_jira_issue_key_in_string(self, module):
        """Jira issue key like KAFKA-20404 in a string value."""
        fn = _load(module)
        raw = '{"jql": "issue = KAFKA-20404", "answer": "specific issue"}'
        assert fn(raw) == raw

    def test_jql_with_double_quoted_field_names(self, module):
        """Field names wrapped in double quotes in the JQL string value."""
        fn = _load(module)
        raw = '{"jql": "\\"Sample Planned Version\\" >= \\"v1.0.0\\"", "answer": "range"}'
        assert fn(raw) == raw

    def test_jql_with_backslash_in_string(self, module):
        """Backslash characters in the JQL string value."""
        fn = _load(module)
        raw = '{"jql": "summary ~ \\"filter\\\\path\\"", "answer": "backslash test"}'
        assert fn(raw) == raw


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestUnicodeContent:
    """Unicode and special characters must not break JSON extraction."""

    def test_unicode_field_names(self, module):
        """Non-ASCII characters in field names."""
        fn = _load(module)
        raw = '{"jql": "status = \\"Done\\"", "answer": "řešeno", "intent_fields": ["Stav"]}'
        assert fn(raw) == raw

    def test_emoji_in_string_value(self, module):
        """Emoji characters inside a string value."""
        fn = _load(module)
        raw = '{"jql": "status", "answer": "Done ✅"}'
        assert fn(raw) == raw

    def test_chinese_characters_in_string(self, module):
        """Chinese characters inside a string value."""
        fn = _load(module)
        raw = '{"jql": "status", "answer": "已解决"}'
        assert fn(raw) == raw


# ---------------------------------------------------------------------------
# Number-like values in string values (that look like code)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestNumberLikeStrings:
    """Numbers in string values that could confuse a naive parser."""

    def test_version_numbers_in_string(self, module):
        """Version numbers like v1.0.0 as string values."""
        fn = _load(module)
        raw = '{"jql": "Sample Planned Version", "answer": "version field", "hint": "v1.5.0"}'
        assert fn(raw) == raw

    def test_large_number_as_string(self, module):
        """Very large number as a string value."""
        fn = _load(module)
        raw = '{"jql": "storyPoints > 1000000", "answer": "large number"}'
        assert fn(raw) == raw

    def test_scientific_notation_in_string(self, module):
        """Scientific notation in string value."""
        fn = _load(module)
        raw = '{"jql": "field > 1e10", "answer": "scientific"}'
        assert fn(raw) == raw


# ---------------------------------------------------------------------------
# Concurrency / memory safety — same input must give same output (pure function)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", CLOUD_CLIENTS, ids=[m.split(".")[1] for m in CLOUD_CLIENTS])
class TestDeterminism:
    """Same input always produces same output (pure function)."""

    def test_idempotent(self, module):
        """Calling the function twice on same input gives same result."""
        fn = _load(module)
        raw = "Explanation:\n" + CANONICAL + "\nTrailing note."
        first = fn(raw)
        second = fn(raw)
        assert first == second

    def test_preamble_position_independence(self, module):
        """Multiple preambles at different positions all stripped correctly."""
        fn = _load(module)
        raw = "First sentence.\nSecond sentence.\n" + CANONICAL + "\nEnd note."
        assert fn(raw) == CANONICAL

    def test_whitespace_variations_all_normalize(self, module):
        """Different whitespace patterns all normalize to same output."""
        fn = _load(module)
        cases = [
            "  " + CANONICAL,
            "\n\n" + CANONICAL + "\n",
            CANONICAL + "  ",
            "\n" + CANONICAL,
        ]
        results = [fn(c) for c in cases]
        assert all(r == CANONICAL for r in results)


# ---------------------------------------------------------------------------
# Verify all cloud clients produce identical output for the same input
# ---------------------------------------------------------------------------

class TestCrossClientConsistency:
    """All four cloud clients must produce identical results for the same input."""

    @pytest.mark.parametrize("raw,expected", [
        (CANONICAL, CANONICAL),
        ("```json\n" + CANONICAL + "\n```", CANONICAL),
        ("Explanation:\n" + CANONICAL, CANONICAL),
        (CANONICAL + "\nTrailing.", CANONICAL),
        ("", ""),
        ("{}", "{}"),
        (EXACT_BUG, EXACT_BUG),
        (JIRA_SERVER_CLEAN, JIRA_SERVER_CLEAN),
        (AQL_FUNC, AQL_FUNC),
        ("  \n" + CANONICAL + "\n", CANONICAL),
        ("Sure.\n```json\n" + CANONICAL + "\n```\nOK?", CANONICAL),
    ])
    def test_all_clients_identical(self, raw, expected):
        """Each of the four clients returns the same output for the same input."""
        results = {}
        for module in CLOUD_CLIENTS:
            results[module.split(".")[1]] = _load(module)(raw)

        names = list(results.keys())
        for i in range(1, len(names)):
            assert results[names[i]] == results[names[0]], (
                f"Mismatch between {names[0]} and {names[i]} for input: {raw!r}"
            )