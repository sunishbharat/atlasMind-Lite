"""
conftest.py — shared fixtures for the aMind-partial test suite.
"""

import json
import pytest

from core.field_resolver import FieldResolver

# ---------------------------------------------------------------------------
# Sample field data (5 fields covering system and custom IDs)
# ---------------------------------------------------------------------------

SAMPLE_JIRA_FIELDS = {
    "customfield_10016": {"name": "Story Points", "type": "number"},
    "customfield_10020": {"name": "Sprint",       "type": "array"},
    "customfield_10014": {"name": "Epic Link",    "type": "string"},
    "priority":          {"name": "Priority",     "type": "option"},
    "assignee":          {"name": "Assignee",     "type": "user"},
}


@pytest.fixture
def sample_name_to_id():
    return {
        "story points": "customfield_10016",
        "sprint":       "customfield_10020",
        "epic link":    "customfield_10014",
        "priority":     "priority",
        "assignee":     "assignee",
    }


@pytest.fixture
def sample_id_to_name():
    return {
        "customfield_10016": "Story Points",
        "customfield_10020": "Sprint",
        "customfield_10014": "Epic Link",
        "priority":          "Priority",
        "assignee":          "Assignee",
    }


@pytest.fixture
def resolver(sample_name_to_id, sample_id_to_name):
    """FieldResolver built from sample DB mappings, capped at 3 intent fields."""
    return FieldResolver.from_db_mappings(
        sample_name_to_id, sample_id_to_name, max_intent_fields=3
    )


@pytest.fixture
def jira_fields_json(tmp_path):
    """Write SAMPLE_JIRA_FIELDS to a temp file and return the Path."""
    p = tmp_path / "jira_fields.json"
    p.write_text(json.dumps(SAMPLE_JIRA_FIELDS), encoding="utf-8")
    return p


@pytest.fixture
def raw_jira_issue():
    """Minimal raw Jira REST API issue dict for normalize_issue tests."""
    return {
        "key": "TEST-1",
        "fields": {
            "summary":        "Fix login bug",
            "description":    "Login fails for SSO users.",
            "status":         {"name": "In Progress"},
            "issuetype":      {"name": "Bug"},
            "priority":       {"name": "High"},
            "assignee":       {"displayName": "Alice"},
            "reporter":       {"displayName": "Bob"},
            "created":        "2024-01-10T09:00:00.000+0000",
            "updated":        "2024-01-11T10:00:00.000+0000",
            "resolutiondate": None,
            "duedate":        None,
            "labels":         ["sso", "login"],
            "comment":        {"comments": []},
            "parent":         None,
        },
    }
