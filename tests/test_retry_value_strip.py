"""
tests/test_retry_value_strip.py

Unit tests for _remove_bad_value_from_in_clause.

All tests are pure-Python — no DB, no LLM, no network.
"""

import pytest
from core.atlasmind import _remove_bad_value_from_in_clause


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _norm(jql: str | None) -> str | None:
    """Collapse runs of whitespace for comparison."""
    return " ".join(jql.split()) if jql is not None else None


# ---------------------------------------------------------------------------
# Returns None (signal to fall back to full-condition strip)
# ---------------------------------------------------------------------------

def test_equality_clause_returns_none():
    """Single-value equality is not an IN clause — caller must strip it."""
    assert _remove_bad_value_from_in_clause(
        "project = Maven AND status = Open", "project", "Maven"
    ) is None


def test_field_not_present_returns_none():
    """Field does not appear in the JQL at all."""
    assert _remove_bad_value_from_in_clause(
        "issuetype = Bug AND status = Open", "project", "Maven"
    ) is None


def test_all_values_bad_returns_none():
    """Every value in the IN list is bad — nothing remains."""
    assert _remove_bad_value_from_in_clause(
        "project IN (Maven) AND status = Open", "project", "Maven"
    ) is None


def test_wrong_field_not_touched():
    """The bad value belongs to a different field — returns None."""
    assert _remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven)", "status", "Maven"
    ) is None


# ---------------------------------------------------------------------------
# Surgical removal — two values in list
# ---------------------------------------------------------------------------

def test_removes_last_value_in_two_value_list():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven) AND priority = Blocker", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA') AND priority = Blocker")


def test_removes_first_value_in_two_value_list():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (Maven, KAFKA) AND priority = Blocker", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA') AND priority = Blocker")


# ---------------------------------------------------------------------------
# Three values in list
# ---------------------------------------------------------------------------

def test_removes_middle_value_in_three_value_list():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven, HIVE) ORDER BY created DESC",
        "project", "Maven",
    ))
    assert result == _norm("project IN ('KAFKA', 'HIVE') ORDER BY created DESC")


def test_removes_first_value_in_three_value_list():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (Maven, KAFKA, HIVE)", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA', 'HIVE')")


def test_removes_last_value_in_three_value_list():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, HIVE, Maven)", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA', 'HIVE')")


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_removal_is_case_insensitive():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, maven)", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA')")


def test_field_match_is_case_insensitive():
    result = _norm(_remove_bad_value_from_in_clause(
        "PROJECT IN (KAFKA, Maven)", "project", "Maven"
    ))
    assert result == _norm("PROJECT IN ('KAFKA')")


# ---------------------------------------------------------------------------
# Pre-quoted values in input
# ---------------------------------------------------------------------------

def test_handles_single_quoted_values():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN ('KAFKA', 'Maven')", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA')")


def test_handles_mixed_quoted_and_unquoted():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN ('KAFKA', Maven)", "project", "Maven"
    ))
    assert result == _norm("project IN ('KAFKA')")


# ---------------------------------------------------------------------------
# NOT IN clause
# ---------------------------------------------------------------------------

def test_not_in_removes_bad_value():
    result = _norm(_remove_bad_value_from_in_clause(
        "status NOT IN (Done, Maven) ORDER BY created DESC",
        "status", "Maven",
    ))
    assert result == _norm("status NOT IN ('Done') ORDER BY created DESC")


def test_not_in_all_values_bad_returns_none():
    assert _remove_bad_value_from_in_clause(
        "status NOT IN (Maven)", "status", "Maven"
    ) is None


# ---------------------------------------------------------------------------
# Surrounding JQL preserved exactly
# ---------------------------------------------------------------------------

def test_preserves_leading_conditions():
    result = _norm(_remove_bad_value_from_in_clause(
        "issuetype = Bug AND project IN (KAFKA, Maven) AND status = Open",
        "project", "Maven",
    ))
    assert result == _norm("issuetype = Bug AND project IN ('KAFKA') AND status = Open")


def test_preserves_order_by():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven) AND priority IN (Blocker, Critical) ORDER BY updated DESC",
        "project", "Maven",
    ))
    assert result == _norm(
        "project IN ('KAFKA') AND priority IN (Blocker, Critical) ORDER BY updated DESC"
    )


def test_only_target_field_modified_other_in_clause_untouched():
    """A second IN clause on a different field must not be altered."""
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven) AND priority IN (Blocker, Critical)",
        "project", "Maven",
    ))
    assert result == _norm("project IN ('KAFKA') AND priority IN (Blocker, Critical)")


# ---------------------------------------------------------------------------
# Leading IN clause (no preceding AND)
# ---------------------------------------------------------------------------

def test_leading_in_clause_no_preceding_and():
    result = _norm(_remove_bad_value_from_in_clause(
        "project IN (KAFKA, Maven) AND priority = Blocker",
        "project", "Maven",
    ))
    assert result == _norm("project IN ('KAFKA') AND priority = Blocker")


# ---------------------------------------------------------------------------
# Real-world shape from the logs
# ---------------------------------------------------------------------------

def test_real_world_kafka_maven_query():
    jql = (
        "project in (KAFKA, Maven) AND priority in (Blocker, Critical) "
        "AND updated >= -90d ORDER BY updated DESC"
    )
    result = _norm(_remove_bad_value_from_in_clause(jql, "project", "Maven"))
    assert result == _norm(
        "project in ('KAFKA') AND priority in (Blocker, Critical) "
        "AND updated >= -90d ORDER BY updated DESC"
    )
