"""
tests/test_parse_limit.py

Verifies _parse_limit extracts the user-specified result count from natural
language queries, including the patterns that previously fell through to the
MAX_RESULTS default and caused the pagination loop to fetch 1000 issues.
"""

import pytest
from core.atlasmind import _parse_limit
from settings import MAX_RESULTS


class TestParseLimitMatchedPatterns:
    """Queries that contain an explicit count — must return that number."""

    @pytest.mark.parametrize("query,expected", [
        # Original keyword-prefix patterns
        ("list 10 issues",                    10),
        ("show 5 tickets",                    5),
        ("get 50 results",                    50),
        ("fetch 100 items",                   100),
        ("top 20 bugs",                       20),
        ("first 15 stories",                  15),
        # New keyword-prefix patterns added in fix
        ("give me 10 issues",                 10),
        ("give 10 issues",                    10),
        ("find me 7 tickets",                 7),
        ("return 30 results",                 30),
        # Suffix-qualified: number immediately before noun
        ("10 issues in project KAFKA",        10),
        ("show 5 issues",                     5),
        # New: adjective between number and noun
        ("give me 10 open issues",            10),
        ("show 20 closed bugs",               20),
        ("list 15 critical tickets",          15),
        ("find 8 unresolved issues",          8),
    ])
    def test_extracts_limit(self, query, expected):
        assert _parse_limit(query) == expected

    def test_negative_lookahead_excludes_time_units(self):
        """Numbers followed by time units must not be treated as result limits."""
        assert _parse_limit("issues created in the last 30 days") == MAX_RESULTS
        assert _parse_limit("updated in the last 7 weeks") == MAX_RESULTS
        assert _parse_limit("first 12 months of the year") == MAX_RESULTS


class TestParseLimitDefaultFallback:
    """Queries with no explicit count must return MAX_RESULTS."""

    @pytest.mark.parametrize("query", [
        "show me open bugs in project KAFKA",
        "list unresolved issues assigned to me",
        "what are the blockers in the current sprint",
        "issues in Driving Functions domain",
    ])
    def test_returns_max_results(self, query):
        assert _parse_limit(query) == MAX_RESULTS
