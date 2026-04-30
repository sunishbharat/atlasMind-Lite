"""
tests/test_jql_sanitizer.py

Unit tests for JqlSanitizer and JiraFieldValueEmbeddings pure logic.

All tests are pure-Python — no DB, no LLM, no network.
JiraFieldValueEmbeddings is mocked via MagicMock where the sanitizer needs it.
Distance values are chosen relative to the defaults in settings.py:
  VALUE_AUTO_CORRECT_THRESHOLD = 0.15   (auto-substitute, high confidence)
  VALUE_HINT_THRESHOLD         = 0.40   (emit hint, medium confidence)
"""

import pytest
from unittest.mock import MagicMock

from core.jql_sanitizer import JqlSanitizer, SanitizeResult, ValueCorrection, ValueHint
from rag.jira_field_value_embeddings import (
    FieldValueRecord,
    JiraFieldValueEmbeddings,
    SimilarValue,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def name_to_id():
    return {
        "customer projects": "customfield_10100",
        "delivery stream":   "customfield_10101",
        "fix version/s":     "fixVersion",
        "status":            "status",
        "issuetype":         "issuetype",
        "assignee":          "assignee",
        "story points":      "customfield_10016",
    }


@pytest.fixture
def id_to_name():
    return {
        "customfield_10100": "Customer Projects",
        "customfield_10101": "Delivery Stream",
        "fixVersion":        "Fix Version/s",
        "status":            "Status",
        "issuetype":         "Issue Type",
        "assignee":          "Assignee",
        "customfield_10016": "Story Points",
    }


@pytest.fixture
def allowed_values():
    return {
        "status":    ["To Do", "In Progress", "In Review", "Done", "Resolved"],
        "issuetype": ["Bug", "Story", "Task", "Epic"],
    }


@pytest.fixture
def mock_fve():
    """JiraFieldValueEmbeddings mock — find_similar_values returns [] by default."""
    mock = MagicMock(spec=JiraFieldValueEmbeddings)
    mock.find_similar_values.return_value = []
    return mock


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def sanitizer(name_to_id, id_to_name, allowed_values, mock_fve, mock_model):
    return JqlSanitizer(
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        allowed_values=allowed_values,
        field_value_embeddings=mock_fve,
        model=mock_model,
    )


# ---------------------------------------------------------------------------
# SanitizeResult model
# ---------------------------------------------------------------------------

class TestSanitizeResult:

    def test_has_hints_false_when_empty(self):
        result = SanitizeResult(jql="status = Done")
        assert result.has_hints is False

    def test_has_hints_true_when_hints_present(self):
        hint = ValueHint(
            field_id="status",
            field_name="Status",
            bad_value="closed",
            candidates=["Done", "Resolved"],
        )
        result = SanitizeResult(jql="", hints=[hint])
        assert result.has_hints is True

    def test_default_corrections_empty(self):
        assert SanitizeResult(jql="").corrections == []

    def test_default_hints_empty(self):
        assert SanitizeResult(jql="").hints == []


# ---------------------------------------------------------------------------
# ValueHint model
# ---------------------------------------------------------------------------

class TestValueHint:

    def test_to_prompt_text_contains_bad_value(self):
        hint = ValueHint(
            field_id="status",
            field_name="Status",
            bad_value="closed",
            candidates=["Done", "Resolved"],
        )
        text = hint.to_prompt_text()
        assert "closed" in text

    def test_to_prompt_text_contains_all_candidates(self):
        hint = ValueHint(
            field_id="status",
            field_name="Status",
            bad_value="closed",
            candidates=["Done", "Resolved", "Won't Fix"],
        )
        text = hint.to_prompt_text()
        assert "Done" in text
        assert "Resolved" in text
        assert "Won't Fix" in text

    def test_to_prompt_text_contains_field_name(self):
        hint = ValueHint(
            field_id="status",
            field_name="Status",
            bad_value="closed",
            candidates=["Done"],
        )
        assert "Status" in hint.to_prompt_text()


# ---------------------------------------------------------------------------
# Phase 1 — field name quoting in WHERE
# ---------------------------------------------------------------------------

class TestPhase1WhereQuoting:

    def test_multiword_field_before_equals_quoted(self, sanitizer):
        result = sanitizer.sanitize('Customer Projects = abc AND status = Done ORDER BY created DESC')
        assert '"Customer Projects"' in result.jql

    def test_multiword_field_before_in_quoted(self, sanitizer):
        result = sanitizer.sanitize('Customer Projects in (abc) ORDER BY created DESC')
        assert '"Customer Projects"' in result.jql

    def test_multiword_field_before_not_in_quoted(self, sanitizer):
        result = sanitizer.sanitize('Customer Projects NOT IN (abc) ORDER BY created DESC')
        assert '"Customer Projects"' in result.jql

    def test_multiword_field_before_is_quoted(self, sanitizer):
        result = sanitizer.sanitize('Customer Projects IS EMPTY ORDER BY created DESC')
        assert '"Customer Projects"' in result.jql

    def test_already_quoted_field_not_double_quoted(self, sanitizer):
        jql = '"Customer Projects" in (abc) ORDER BY created DESC'
        result = sanitizer.sanitize(jql)
        assert '""Customer Projects""' not in result.jql
        assert '"Customer Projects"' in result.jql

    def test_single_word_field_not_quoted(self, sanitizer):
        result = sanitizer.sanitize('status = Done ORDER BY created DESC')
        assert '"status"' not in result.jql

    def test_alphanumeric_multiword_field_quoted(self, sanitizer):
        result = sanitizer.sanitize('Delivery Stream in (abc) ORDER BY created DESC')
        assert '"Delivery Stream"' in result.jql

    def test_field_in_value_string_not_quoted(self, sanitizer):
        # "Customer Projects" appears inside a quoted value — must not be double-quoted
        jql = 'summary ~ "Customer Projects issue" ORDER BY created DESC'
        result = sanitizer.sanitize(jql)
        assert '""Customer Projects""' not in result.jql

    def test_longest_field_matched_first(self, sanitizer, name_to_id, id_to_name, allowed_values, mock_fve, mock_model):
        # Add a name that is a prefix of another to verify longest-first ordering
        name_to_id["customer"] = "customfield_10200"
        id_to_name["customfield_10200"] = "Customer"
        s = JqlSanitizer(
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            allowed_values=allowed_values,
            field_value_embeddings=mock_fve,
            model=mock_model,
        )
        result = s.sanitize('"Customer Projects" in (abc) ORDER BY created DESC')
        # Already-quoted "Customer Projects" should remain as-is — no re-quoting
        assert result.jql.count('"Customer Projects"') == 1
        assert '""' not in result.jql


# ---------------------------------------------------------------------------
# Phase 1 — field name quoting in ORDER BY
# ---------------------------------------------------------------------------

class TestPhase1OrderByQuoting:

    def test_multiword_field_in_orderby_quoted(self, sanitizer):
        result = sanitizer.sanitize('status = Done ORDER BY Customer Projects ASC')
        assert '"Customer Projects"' in result.jql

    def test_multiword_field_in_orderby_desc_quoted(self, sanitizer):
        result = sanitizer.sanitize('status = Done ORDER BY Customer Projects DESC')
        assert '"Customer Projects"' in result.jql

    def test_already_quoted_orderby_not_double_quoted(self, sanitizer):
        jql = 'status = Done ORDER BY "Customer Projects" ASC'
        result = sanitizer.sanitize(jql)
        assert '""Customer Projects""' not in result.jql

    def test_single_word_orderby_not_quoted(self, sanitizer):
        result = sanitizer.sanitize('status = Done ORDER BY created DESC')
        assert '"created"' not in result.jql


# ---------------------------------------------------------------------------
# Pass 2 — multi-word IN value quoting
# ---------------------------------------------------------------------------

class TestMultiwordInValueQuoting:

    def test_multiword_value_in_clause_quoted(self, sanitizer):
        result = sanitizer.sanitize(
            'status = Done AND issuetype in (Requirements Change Request) ORDER BY created DESC'
        )
        assert '"Requirements Change Request"' in result.jql

    def test_single_word_value_not_quoted(self, sanitizer):
        result = sanitizer.sanitize('issuetype in (Bug) ORDER BY created DESC')
        assert '"Bug"' not in result.jql

    def test_already_quoted_value_not_double_quoted(self, sanitizer, allowed_values):
        # "Bug" is already quoted AND is a valid issuetype value
        result = sanitizer.sanitize('issuetype in ("Bug") ORDER BY created DESC')
        assert '""Bug""' not in result.jql

    def test_multiple_values_mixed_quoting(self, sanitizer):
        result = sanitizer.sanitize(
            'status = Done AND issuetype in (Bug, Design Change Request) ORDER BY created DESC'
        )
        assert '"Design Change Request"' in result.jql
        # Single-word "Bug" should not gain extra quotes
        assert result.jql.count('"Bug"') <= 1


# ---------------------------------------------------------------------------
# Structural passes
# ---------------------------------------------------------------------------

class TestStructuralPasses:

    def test_limit_stripped(self, sanitizer):
        result = sanitizer.sanitize('status = Done ORDER BY created DESC LIMIT 10')
        assert 'LIMIT' not in result.jql.upper()

    def test_arithmetic_orderby_stripped(self, sanitizer):
        result = sanitizer.sanitize(
            'status = Done ORDER BY resolutiondate - created DESC'
        )
        assert 'ORDER BY resolutiondate' not in result.jql

    def test_numeric_value_dequoted(self, sanitizer):
        result = sanitizer.sanitize("Sprint in ('224') ORDER BY created DESC")
        assert "'224'" not in result.jql
        assert "224" in result.jql


# ---------------------------------------------------------------------------
# Phase 2 — value validation: exact match
# ---------------------------------------------------------------------------

class TestPhase2ExactMatch:

    def test_exact_match_kept(self, sanitizer):
        result = sanitizer.sanitize(
            'status = Done AND issuetype = Bug ORDER BY created DESC'
        )
        assert 'Done' in result.jql
        assert 'Bug' in result.jql
        assert result.corrections == []
        assert result.hints == []

    def test_case_insensitive_match_normalised(self, sanitizer):
        result = sanitizer.sanitize(
            'status = done ORDER BY created DESC'
        )
        # Normalised to canonical casing from Jira server
        assert 'Done' in result.jql
        assert 'done' not in result.jql

    def test_case_insensitive_in_clause_normalised(self, sanitizer):
        result = sanitizer.sanitize(
            'issuetype in (bug, STORY) ORDER BY created DESC'
        )
        assert 'Bug' in result.jql
        assert 'Story' in result.jql
        assert 'bug' not in result.jql
        assert 'STORY' not in result.jql

    def test_free_text_field_skipped(self, sanitizer):
        # assignee is not in allowed_values → pass through unchanged
        result = sanitizer.sanitize(
            'status = Done AND assignee = "john.doe" ORDER BY created DESC'
        )
        assert 'john.doe' in result.jql
        assert result.corrections == []
        assert result.hints == []


# ---------------------------------------------------------------------------
# Phase 2 — value validation: auto-correct (high confidence)
# ---------------------------------------------------------------------------

class TestPhase2AutoCorrect:

    def test_high_confidence_match_auto_corrected(self, sanitizer, mock_fve):
        # Distance 0.10 < AUTO_CORRECT_THRESHOLD (0.15) → auto-substitute
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="In Progress", distance=0.10),
        ]
        result = sanitizer.sanitize(
            'status = "In Progres" ORDER BY created DESC'
        )
        assert "In Progress" in result.jql
        assert "In Progres" not in result.jql
        assert len(result.corrections) == 1
        assert result.corrections[0].original_value == "In Progres"
        assert result.corrections[0].corrected_value == "In Progress"
        assert result.hints == []

    def test_auto_correction_records_distance(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.08),
        ]
        result = sanitizer.sanitize('status = "Dne" ORDER BY created DESC')
        assert result.corrections[0].distance == pytest.approx(0.08)

    def test_auto_correction_records_field_name(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.08),
        ]
        result = sanitizer.sanitize('status = "Dne" ORDER BY created DESC')
        assert result.corrections[0].field_name == "Status"


# ---------------------------------------------------------------------------
# Phase 2 — value validation: hint (medium confidence)
# ---------------------------------------------------------------------------

class TestPhase2Hint:

    def test_medium_confidence_emits_hint(self, sanitizer, mock_fve):
        # Distance 0.25 >= AUTO_CORRECT_THRESHOLD but < HINT_THRESHOLD (0.40)
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.25),
            SimilarValue(value="Resolved", distance=0.30),
        ]
        result = sanitizer.sanitize('status = "closed" ORDER BY created DESC')
        assert result.has_hints is True
        assert len(result.hints) == 1
        hint = result.hints[0]
        assert hint.bad_value == "closed"
        assert "Done" in hint.candidates
        assert "Resolved" in hint.candidates

    def test_hint_condition_stripped_from_jql(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.25),
        ]
        result = sanitizer.sanitize('status = "closed" ORDER BY created DESC')
        assert "closed" not in result.jql

    def test_hint_field_name_populated(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.25),
        ]
        result = sanitizer.sanitize('status = "closed" ORDER BY created DESC')
        assert result.hints[0].field_name == "Status"

    def test_only_candidates_below_hint_threshold_included(self, sanitizer, mock_fve):
        # First candidate below threshold, second above → only first in hint
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.25),
            SimilarValue(value="Resolved", distance=0.60),   # above HINT_THRESHOLD
        ]
        result = sanitizer.sanitize('status = "closed" ORDER BY created DESC')
        assert result.hints[0].candidates == ["Done"]


# ---------------------------------------------------------------------------
# Phase 2 — value validation: strip (no candidates)
# ---------------------------------------------------------------------------

class TestPhase2Strip:

    def test_no_candidates_strips_condition(self, sanitizer, mock_fve):
        # All similarity distances above HINT_THRESHOLD → strip silently
        mock_fve.find_similar_values.return_value = [
            SimilarValue(value="Done", distance=0.70),
        ]
        result = sanitizer.sanitize('status = "xyz" ORDER BY created DESC')
        assert "xyz" not in result.jql
        assert result.hints == []
        assert result.corrections == []

    def test_empty_similarity_results_strips_condition(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = []
        result = sanitizer.sanitize('status = "xyz" ORDER BY created DESC')
        assert "xyz" not in result.jql

    def test_in_clause_all_invalid_stripped(self, sanitizer, mock_fve):
        mock_fve.find_similar_values.return_value = []
        result = sanitizer.sanitize(
            'status = Done AND issuetype in (FakeType) ORDER BY created DESC'
        )
        assert "FakeType" not in result.jql
        assert "issuetype" not in result.jql

    def test_in_clause_partial_valid_partial_invalid(self, sanitizer, mock_fve):
        # Bug is valid (exact match after normalise); FakeType has no candidates
        mock_fve.find_similar_values.return_value = []
        result = sanitizer.sanitize(
            'status = Done AND issuetype in (Bug, FakeType) ORDER BY created DESC'
        )
        assert "Bug" in result.jql
        assert "FakeType" not in result.jql


# ---------------------------------------------------------------------------
# Full sanitize() — combined passes
# ---------------------------------------------------------------------------

class TestSanitizeFull:

    def test_multiword_field_and_in_value_both_fixed(self, sanitizer):
        # Both field name quoting and IN value quoting applied
        result = sanitizer.sanitize(
            'Customer Projects in (abcd ddcc) ORDER BY created DESC'
        )
        assert '"Customer Projects"' in result.jql
        assert '"abcd ddcc"' in result.jql

    def test_valid_jql_passes_through_unchanged(self, sanitizer):
        jql = 'status = Done AND issuetype = Bug ORDER BY created DESC'
        result = sanitizer.sanitize(jql)
        assert "Done" in result.jql
        assert "Bug" in result.jql
        assert result.corrections == []
        assert result.hints == []

    def test_limit_and_field_quoting_both_applied(self, sanitizer):
        result = sanitizer.sanitize(
            'Customer Projects in (abc) ORDER BY created DESC LIMIT 20'
        )
        assert '"Customer Projects"' in result.jql
        assert 'LIMIT' not in result.jql.upper()

    def test_corrections_and_hints_independent(self, sanitizer, mock_fve):
        # status has a typo (auto-correct), issuetype has ambiguous value (hint)
        def side_effect(field_id, bad_value, model, top_n=3):
            if field_id == "status":
                return [SimilarValue(value="In Progress", distance=0.10)]
            if field_id == "issuetype":
                return [SimilarValue(value="Bug", distance=0.25)]
            return []
        mock_fve.find_similar_values.side_effect = side_effect

        result = sanitizer.sanitize(
            'status = "In Progres" AND issuetype = "closed issue" ORDER BY created DESC'
        )
        assert len(result.corrections) == 1
        assert result.corrections[0].corrected_value == "In Progress"
        assert len(result.hints) == 1
        assert result.hints[0].bad_value == "closed issue"


# ---------------------------------------------------------------------------
# JiraFieldValueEmbeddings — pure logic (no DB)
# ---------------------------------------------------------------------------

class TestFieldValueRecord:

    def test_record_fields(self):
        r = FieldValueRecord(field_id="status", field_name="Status", value="Done")
        assert r.field_id == "status"
        assert r.field_name == "Status"
        assert r.value == "Done"


class TestSimilarValue:

    def test_similar_value_fields(self):
        sv = SimilarValue(value="Done", distance=0.12)
        assert sv.value == "Done"
        assert sv.distance == pytest.approx(0.12)


class TestBuildRecords:

    def test_builds_one_record_per_value(self):
        allowed = {"status": ["To Do", "Done"], "issuetype": ["Bug"]}
        id_to_name = {"status": "Status", "issuetype": "Issue Type"}
        records = JiraFieldValueEmbeddings._build_records(allowed, id_to_name)
        assert len(records) == 3

    def test_field_name_resolved_from_id_to_name(self):
        allowed = {"status": ["Done"]}
        id_to_name = {"status": "Status"}
        records = JiraFieldValueEmbeddings._build_records(allowed, id_to_name)
        assert records[0].field_name == "Status"

    def test_unknown_field_id_falls_back_to_id(self):
        allowed = {"customfield_99": ["Value"]}
        records = JiraFieldValueEmbeddings._build_records(allowed, {})
        assert records[0].field_name == "customfield_99"

    def test_empty_allowed_values_returns_empty(self):
        records = JiraFieldValueEmbeddings._build_records({}, {})
        assert records == []

    def test_record_values_match_input(self):
        allowed = {"status": ["To Do", "In Progress", "Done"]}
        id_to_name = {"status": "Status"}
        records = JiraFieldValueEmbeddings._build_records(allowed, id_to_name)
        values = [r.value for r in records]
        assert values == ["To Do", "In Progress", "Done"]

    def test_all_records_have_correct_field_id(self):
        allowed = {"status": ["To Do", "Done"]}
        records = JiraFieldValueEmbeddings._build_records(allowed, {"status": "Status"})
        assert all(r.field_id == "status" for r in records)
