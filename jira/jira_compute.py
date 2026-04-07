"""
jira_compute.py — Derived field calculations from raw Jira issue data.

Functions here operate on the fields returned by the Jira REST API and produce
computed values (e.g. effort in days/hours) that are not available as native
Jira fields. All functions are pure: they take raw field values and return
computed results with no side-effects.
"""

from datetime import datetime, timezone


def parse_jira_dt(value: str | None) -> datetime | None:
    """Parse a Jira ISO-8601 timestamp string into a UTC-aware datetime.

    Jira returns offsets in the form +0000 which Python's fromisoformat does
    not accept before 3.11; this function normalises them to +00:00 first.

    Args:
        value: Timestamp string from the Jira API, e.g. "2024-03-01T10:00:00.000+0000".

    Returns:
        UTC-aware datetime, or None if the value is missing or cannot be parsed.
    """
    if not value:
        return None
    try:
        normalised = value.replace("+0000", "+00:00")
        dt = datetime.fromisoformat(normalised)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def compute_effort(
    created: str | None,
    resolutiondate: str | None,
) -> tuple[float | None, float | None]:
    """Calculate the elapsed time between issue creation and resolution.

    For resolved issues this is the full cycle time.
    Returns (None, None) for open issues (no resolutiondate) or when either
    timestamp is missing or unparseable.

    Args:
        created: Jira 'created' field value (ISO-8601 string).
        resolutiondate: Jira 'resolutiondate' field value (ISO-8601 string).

    Returns:
        (effort_days, effort_hours) both rounded to 2 decimal places, or
        (None, None) when either timestamp is absent.
    """
    start = parse_jira_dt(created)
    end   = parse_jira_dt(resolutiondate)
    if start is None or end is None:
        return None, None
    delta_seconds = (end - start).total_seconds()
    return round(delta_seconds / 86400, 2), round(delta_seconds / 3600, 2)


def compute_age(created: str | None) -> float | None:
    """Calculate how many days ago an issue was created (relative to now, UTC).

    Useful for open issues where there is no resolution date.

    Args:
        created: Jira 'created' field value (ISO-8601 string).

    Returns:
        Age in days (float, 2 d.p.), or None if the timestamp is missing.
    """
    start = parse_jira_dt(created)
    if start is None:
        return None
    delta_seconds = (datetime.now(timezone.utc) - start).total_seconds()
    return round(delta_seconds / 86400, 2)


def compute_time_in_status(
    status_changed: str | None,
    reference: str | None = None,
) -> float | None:
    """Calculate days spent in the current status.

    Args:
        status_changed: ISO-8601 timestamp when the issue last changed status.
        reference: Upper-bound timestamp (defaults to now UTC when omitted).

    Returns:
        Days in current status (float, 2 d.p.), or None if data is missing.
    """
    start = parse_jira_dt(status_changed)
    if start is None:
        return None
    end = parse_jira_dt(reference) if reference else datetime.now(timezone.utc)
    delta_seconds = (end - start).total_seconds()
    return round(delta_seconds / 86400, 2)


def enrich_issue(issue_fields: dict) -> dict:
    """Return a dict of all computed fields derived from a raw Jira fields dict.

    Intended to be merged into the normalised issue dict produced by
    normalize_issue() in core/atlasmind.py.

    Args:
        issue_fields: The 'fields' sub-dict from a raw Jira API issue object.

    Returns:
        dict with keys: effort_days, effort_hours, age_days.
        Values are float or None.
    """
    created        = issue_fields.get("created")
    resolutiondate = issue_fields.get("resolutiondate")

    effort_days, effort_hours = compute_effort(created, resolutiondate)

    return {
        "effort_days":  effort_days,
        "effort_hours": effort_hours,
        "age_days":     compute_age(created) if effort_days is None else None,
    }
