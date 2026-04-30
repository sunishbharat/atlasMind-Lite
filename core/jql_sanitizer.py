"""
jql_sanitizer.py — Deterministic JQL cleanup and value correction.

JqlSanitizer applies all post-generation cleanup passes to LLM-produced JQL:

  Phase 1 — Field name quoting (DB-backed, field-position aware):
    Detects known multi-word field names in WHERE and ORDER BY positions
    and wraps them in double quotes. Uses name_to_id from the vector DB as
    ground truth — no guessing, no false positives on value strings.

  Phase 2 — Value validation and correction (exact → similarity → hint):
    For each field condition, validates the value against the known allowed
    values fetched from the Jira server:
      - Exact match (case-insensitive) → normalise to canonical casing.
      - No exact match → cosine similarity search in jira_field_values:
          distance < AUTO_CORRECT_THRESHOLD → auto-substitute (obvious typo).
          distance < HINT_THRESHOLD         → emit ValueHint for retry prompt.
          no candidates                     → strip condition, log warning.

  Passes 3-7 — Structural cleanup:
    Strip LIMIT, strip arithmetic ORDER BY, strip field-to-field date arithmetic,
    dequote numeric values, quote multi-word IN values.

All passes are deterministic — no LLM call, zero token cost.
"""

import logging
import re

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from rag.jira_field_value_embeddings import JiraFieldValueEmbeddings
from settings import (
    VALUE_AUTO_CORRECT_THRESHOLD,
    VALUE_HINT_MAX_CANDIDATES,
    VALUE_HINT_THRESHOLD,
)

logger = logging.getLogger(__name__)

# JQL operators that follow a field name in a WHERE condition.
_JQL_WHERE_OPERATORS = (
    r"=|!=|<=?|>=?|~|!~"
    r"|\bIN\b|\bNOT\s+IN\b|\bIS\s+NOT\b|\bIS\b|\bWAS\s+NOT\b|\bWAS\b|\bCHANGED\b"
)

_JQL_QUOTED_NUMBER_RE = re.compile(r"""(['"])(\d+)\1""")
_JQL_IN_CLAUSE_RE = re.compile(r"((?:NOT\s+)?IN)\s*\(([^)]*)\)", re.IGNORECASE)
_JQL_LIMIT_RE = re.compile(r"\s+LIMIT\s+\d+", re.IGNORECASE)
_JQL_ARITHMETIC_ORDER_RE = re.compile(
    r"\s+ORDER\s+BY\s+\S+\s*[-+]\s*.*$", re.IGNORECASE
)
_JQL_FIELD_ARITH_COND_RE = re.compile(
    r"\s+AND\s+"
    r"(?:"
    r"\w+\s*(?:>=?|<=?|!=?)\s*[a-zA-Z]\w*\s*[-+]\s*\d+[a-zA-Z]+"
    r"|"
    r"\w+\s*[-+]\s*\d+[a-zA-Z]+\s*(?:>=?|<=?|!=?)\s*\w+"
    r"|"
    r"\w+\s*[-+]\s*\w+\s*(?:>=?|<=?|!=?)\s*\d+[a-zA-Z]+"
    r")",
    re.IGNORECASE,
)
# Matches AND field = 'value' or AND field != 'value'.
_JQL_AND_EQUALITY_RE = re.compile(
    r"""\s+AND\s+([\w\[\]]+)\s*(!=|=)\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)
# Matches AND field IN (...) or AND field NOT IN (...).
_JQL_AND_IN_RE = re.compile(
    r"""\s+AND\s+([\w\[\]]+)\s+(NOT\s+IN|IN)\s*\(([\s\S]*?)\)""",
    re.IGNORECASE,
)


class ValueCorrection(BaseModel):
    """An auto-corrected value substitution made by the sanitizer (high confidence)."""
    field_id: str
    field_name: str
    original_value: str
    corrected_value: str
    distance: float


class ValueHint(BaseModel):
    """A correction hint for the LLM retry prompt (medium confidence).

    Emitted when the sanitizer found close candidates but not a single
    unambiguous winner — the LLM resolves the ambiguity using query intent.
    """
    field_id: str
    field_name: str
    bad_value: str
    candidates: list[str]

    def to_prompt_text(self) -> str:
        """Format as a compact sentence for injection into the retry prompt."""
        candidates_str = ", ".join(self.candidates)
        return (
            f"Value '{self.bad_value}' is not valid for field '{self.field_name}'. "
            f"Closest valid values: {candidates_str}. "
            f"Use one of these or remove the condition."
        )


class SanitizeResult(BaseModel):
    """Complete output of a JqlSanitizer.sanitize() call."""
    jql: str
    corrections: list[ValueCorrection] = []
    hints: list[ValueHint] = []

    @property
    def has_hints(self) -> bool:
        return len(self.hints) > 0


class JqlSanitizer:
    """Applies deterministic cleanup and DB-backed value correction to LLM JQL.

    Constructed once in AtlasMind.run() after field mappings and allowed values
    are loaded from the vector DB. All state is immutable after construction.

    Args:
        name_to_id:             {field_name.lower(): field_id} from FieldResolver.
        id_to_name:             {field_id: display_name} from FieldResolver.
        allowed_values:         {field_id: [canonical_value, ...]} from fetch_allowed_values().
        field_value_embeddings: JiraFieldValueEmbeddings for cosine similarity search.
        model:                  SentenceTransformer — same model used at seeding time.
    """

    def __init__(
        self,
        name_to_id: dict[str, str],
        id_to_name: dict[str, str],
        allowed_values: dict[str, list[str]],
        field_value_embeddings: JiraFieldValueEmbeddings,
        model: SentenceTransformer,
    ) -> None:
        self._id_to_name = id_to_name
        self._allowed_values = allowed_values
        self._field_value_embeddings = field_value_embeddings
        self._model = model

        # Normalised allowed values: {field_id_lower: {value_lower: canonical_value}}
        # Used for O(1) exact lookup and casing normalisation.
        self._normed: dict[str, dict[str, str]] = {
            fid.lower(): {v.lower(): v for v in vals}
            for fid, vals in allowed_values.items()
        }

        # Pre-compile WHERE and ORDER BY patterns for each known multi-word field name.
        # Sorted longest-first so "Customer Projects Group" is matched before
        # "Customer Projects", preventing partial double-quoting.
        self._field_patterns: list[tuple[str, re.Pattern, re.Pattern]] = (
            self._build_field_patterns(name_to_id)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sanitize(self, jql: str) -> SanitizeResult:
        """Apply all cleanup passes to raw LLM-produced JQL.

        Returns:
            SanitizeResult containing the cleaned JQL string, any auto-corrections
            made by the sanitizer (logged, not visible to the user), and any
            ValueHints to inject into the next retry prompt.
        """
        corrections: list[ValueCorrection] = []
        hints: list[ValueHint] = []

        # Pass 1 — quote known multi-word field names in WHERE and ORDER BY.
        jql = self._quote_field_names(jql)

        # Pass 2 — quote multi-word values inside IN (...) clauses.
        quoted = self._quote_multiword_in_values(jql)
        if quoted != jql:
            logger.info("JQL after multi-word IN-value quoting: %s", quoted)
        jql = quoted

        # Pass 3 — strip LIMIT clause (not supported by Jira REST API).
        clean = _JQL_LIMIT_RE.sub("", jql).strip()
        if clean != jql:
            logger.info("JQL after LIMIT strip: %s", clean)
        jql = clean

        # Pass 4 — strip arithmetic in ORDER BY (e.g. ORDER BY resolutiondate - created).
        stripped = _JQL_ARITHMETIC_ORDER_RE.sub("", jql).strip()
        if stripped != jql:
            logger.warning("JQL: arithmetic in ORDER BY stripped: %s", jql)
        jql = stripped

        # Pass 5 — strip field-to-field date arithmetic in WHERE conditions.
        stripped = _JQL_FIELD_ARITH_COND_RE.sub("", jql).strip()
        if stripped != jql:
            logger.warning("JQL: field-to-field date arithmetic in WHERE stripped: %s", jql)
        jql = stripped

        # Pass 6 — dequote purely numeric values (e.g. Sprint in ('224') → Sprint in (224)).
        unquoted = _JQL_QUOTED_NUMBER_RE.sub(r"\2", jql)
        if unquoted != jql:
            logger.info("JQL after numeric dequote: %s", unquoted)
        jql = unquoted

        # Pass 7 — validate values against allowed sets; correct or emit hints.
        jql, corrections, hints = self._validate_and_correct_values(jql)

        return SanitizeResult(jql=jql, corrections=corrections, hints=hints)

    # ------------------------------------------------------------------
    # Phase 1 — field name quoting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_field_patterns(
        name_to_id: dict[str, str],
    ) -> list[tuple[str, re.Pattern, re.Pattern]]:
        """Pre-compile WHERE and ORDER BY match patterns for each multi-word field name.

        Returns a list of (canonical_name, where_pattern, order_pattern) triples
        sorted longest-first so longer names are matched before shorter substrings.
        """
        multiword_names = sorted(
            (name for name in name_to_id if " " in name),
            key=len,
            reverse=True,
        )
        patterns: list[tuple[str, re.Pattern, re.Pattern]] = []
        for name in multiword_names:
            escaped = re.escape(name)
            # WHERE pattern: match the field name when:
            #   - not already preceded by a double-quote (not already quoted)
            #   - followed by optional whitespace then a JQL operator
            where_pat = re.compile(
                rf'(?<!")({escaped})(?!")(?=\s*(?:{_JQL_WHERE_OPERATORS}))',
                re.IGNORECASE,
            )
            # ORDER BY pattern: match the field name when not already quoted.
            # Applied only to the ORDER BY portion of the JQL string so there
            # is no risk of matching values in the WHERE clause.
            order_pat = re.compile(
                rf'(?<!")({escaped})(?!")',
                re.IGNORECASE,
            )
            patterns.append((name, where_pat, order_pat))

        logger.info(
            "JqlSanitizer: compiled patterns for %d multi-word field names.",
            len(patterns),
        )
        return patterns

    def _quote_field_names(self, jql: str) -> str:
        """Quote unquoted multi-word field names in WHERE and ORDER BY positions."""
        order_match = re.search(r"\bORDER\s+BY\b", jql, re.IGNORECASE)
        if order_match:
            where_part = jql[: order_match.start()]
            order_part = jql[order_match.start() :]
        else:
            where_part = jql
            order_part = ""

        for name, where_pat, order_pat in self._field_patterns:
            quoted_name = f'"{name}"'

            new_where = where_pat.sub(quoted_name, where_part)
            if new_where != where_part:
                logger.info("JQL: quoted multi-word field %r in WHERE.", name)
            where_part = new_where

            if order_part:
                new_order = order_pat.sub(quoted_name, order_part)
                if new_order != order_part:
                    logger.info("JQL: quoted multi-word field %r in ORDER BY.", name)
                order_part = new_order

        return where_part + order_part

    # ------------------------------------------------------------------
    # Pass 2 — multi-word value quoting in IN clauses
    # ------------------------------------------------------------------

    @staticmethod
    def _quote_multiword_in_values(jql: str) -> str:
        """Quote unquoted multi-word values inside IN (...) clauses."""
        def _fix(m: re.Match) -> str:
            in_kw = m.group(1)
            parts = [p.strip() for p in m.group(2).split(",") if p.strip()]
            fixed: list[str] = []
            for p in parts:
                if (p.startswith('"') and p.endswith('"')) or (
                    p.startswith("'") and p.endswith("'")
                ):
                    fixed.append(p)
                elif " " in p:
                    fixed.append(f'"{p}"')
                else:
                    fixed.append(p)
            return f"{in_kw} ({', '.join(fixed)})"

        return _JQL_IN_CLAUSE_RE.sub(_fix, jql)

    # ------------------------------------------------------------------
    # Phase 2 — value validation and correction
    # ------------------------------------------------------------------

    def _validate_and_correct_values(
        self, jql: str
    ) -> tuple[str, list[ValueCorrection], list[ValueHint]]:
        """Validate field values against allowed sets; correct or emit hints."""
        if not self._allowed_values:
            return jql, [], []

        corrections: list[ValueCorrection] = []
        hints: list[ValueHint] = []

        def _resolve_value(
            field_raw: str,
            value: str,
        ) -> tuple[str | None, ValueCorrection | None, ValueHint | None]:
            """Return (corrected_value | None, correction | None, hint | None).

            corrected_value=None signals that the condition should be stripped.
            When hint is returned the condition is also stripped — the LLM
            rewrites the condition in the next retry using the candidate list.
            """
            field_key = field_raw.lower()
            value_map = self._normed.get(field_key)

            if value_map is None:
                return value, None, None  # free-text / date / number — skip validation

            # Exact match: normalise to canonical casing from the Jira server.
            if value.lower() in value_map:
                canonical = value_map[value.lower()]
                if canonical != value:
                    logger.info(
                        "JQL: normalised value %r → %r for field %r",
                        value, canonical, field_raw,
                    )
                return canonical, None, None

            # No exact match — cosine similarity search (zero LLM tokens).
            field_name = self._id_to_name.get(field_key, field_raw)
            similar = self._field_value_embeddings.find_similar_values(
                field_id=field_key,
                bad_value=value,
                model=self._model,
                top_n=VALUE_HINT_MAX_CANDIDATES,
            )

            if similar and similar[0].distance < VALUE_AUTO_CORRECT_THRESHOLD:
                best = similar[0]
                correction = ValueCorrection(
                    field_id=field_key,
                    field_name=field_name,
                    original_value=value,
                    corrected_value=best.value,
                    distance=best.distance,
                )
                logger.info(
                    "JQL: auto-corrected %r → %r for field %r (distance=%.3f)",
                    value, best.value, field_raw, best.distance,
                )
                return best.value, correction, None

            close_candidates = [s.value for s in similar if s.distance < VALUE_HINT_THRESHOLD]
            if close_candidates:
                hint = ValueHint(
                    field_id=field_key,
                    field_name=field_name,
                    bad_value=value,
                    candidates=close_candidates,
                )
                logger.warning(
                    "JQL: value %r invalid for field %r — emitting hint: %s",
                    value, field_raw, close_candidates,
                )
                return None, None, hint

            logger.warning(
                "JQL: value %r invalid for field %r, no close candidates — stripping condition.",
                value, field_raw,
            )
            return None, None, None

        def _sub_equality(m: re.Match) -> str:
            field = m.group(1)
            op = m.group(2)
            value = m.group(3)
            corrected, correction, hint = _resolve_value(field, value)
            if corrected is None:
                if hint:
                    hints.append(hint)
                return ""
            if correction:
                corrections.append(correction)
            return f" AND {field} {op} '{corrected}'"

        def _sub_in(m: re.Match) -> str:
            field = m.group(1)
            in_keyword = m.group(2)
            field_key = field.lower()
            if field_key not in self._normed:
                return m.group(0)

            raw_values = [v.strip().strip("'\"") for v in m.group(3).split(",") if v.strip()]
            valid_values: list[str] = []
            for v in raw_values:
                corrected, correction, hint = _resolve_value(field, v)
                if corrected is not None:
                    valid_values.append(corrected)
                    if correction:
                        corrections.append(correction)
                elif hint:
                    hints.append(hint)

            if not valid_values:
                return ""
            values_str = ", ".join(f"'{v}'" for v in valid_values)
            return f" AND {field} {in_keyword} ({values_str})"

        result = _JQL_AND_EQUALITY_RE.sub(_sub_equality, jql)
        result = _JQL_AND_IN_RE.sub(_sub_in, result)
        return result.strip(), corrections, hints
