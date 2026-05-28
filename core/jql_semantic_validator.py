"""
jql_semantic_validator.py — Post-generation JQL field/value correction via embedding similarity.

Runs between LLM generation and JqlSanitizer. Uses the LLM's own clauses[] output (or
a regex fallback) to correct field names and values before the query reaches Jira.
Zero extra LLM calls — pure embedding lookups against existing pgvector stores.
"""

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from core.models import LlmClause
from pgvector_client import PGVectorClient
from settings import (
    JIRA_FIELD_TABLE,
    JIRA_FIELD_COL_EMBEDDING,
    VALUE_AUTO_CORRECT_THRESHOLD,
    SEMANTIC_FIELD_THRESHOLD,
)

if TYPE_CHECKING:
    from rag.jira_field_embeddings import Jira_Field_Embeddings
    from rag.jira_field_value_embeddings import JiraFieldValueEmbeddings
    from rag.jira_asset_embeddings import JiraAssetEmbeddings

logger = logging.getLogger(__name__)

# Operators that require numeric or date field types
_NUMERIC_DATE_OPS = {">", "<", ">=", "<="}
# Operators that require text-searchable field types
_TEXT_OPS = {"~", "!~"}
# Field types compatible with numeric/date comparison operators
_NUMERIC_DATE_TYPES = {"number", "date", "datetime", "priority"}
# Field types compatible with text search operators
_TEXT_TYPES = {"string", "text", "comments-page"}

# Regex patterns for the fallback clause parser
# Matches: field op 'value' or field op "value"
_RE_SINGLE_CLAUSE = re.compile(
    r"""(?P<field>[A-Za-z][A-Za-z0-9_\- ]*)"""
    r"""\s+(?P<op>!=|!~|NOT\s+IN|IN|>=|<=|~|=|>|<|IS\s+NOT|IS)"""
    r"""\s+(?:'(?P<sq>[^']*)'|"(?P<dq>[^"]*)")""",
    re.IGNORECASE,
)
# Matches: field IN ('a', 'b', ...)
_RE_IN_CLAUSE = re.compile(
    r"""(?P<field>[A-Za-z][A-Za-z0-9_\- ]*)\s+(?P<op>NOT\s+IN|IN)\s*\((?P<vals>[^)]+)\)""",
    re.IGNORECASE,
)
# Matches aqlFunction — always skip
_RE_AQL = re.compile(r"aqlFunction\s*\(", re.IGNORECASE)
# Matches JQL functions — value not correctable
_RE_FUNCTION = re.compile(r"^\w+\s*\(", re.IGNORECASE)
# Matches relative date expressions like -30d, -4w, -12M
_RE_DATE_EXPR = re.compile(r"^-?\d+[dwM]$")
# Matches issue keys like ABCD-1234
_RE_ISSUE_KEY = re.compile(r"^[A-Z]+-\d+$")


class JqlClause(BaseModel):
    """Internal representation of one correctable clause."""
    field_name: str
    operator:   str
    values:     list[str]
    raw_text:   str
    correctable: bool


class FieldCorrection(BaseModel):
    original_name: str
    field_id:      str
    display_name:  str
    distance:      float
    type_compatible: bool


class SemanticValueCorrection(BaseModel):
    original_value:  str
    corrected_value: str
    distance:        float
    source:          str   # "standard" or "asset"


class ValidatedClause(BaseModel):
    original:          JqlClause
    field_correction:  FieldCorrection | None
    value_corrections: list[SemanticValueCorrection]
    rewritten_text:    str


class ValidatedJql(BaseModel):
    original_jql:  str
    corrected_jql: str
    clauses:       list[ValidatedClause]

    @property
    def was_modified(self) -> bool:
        return self.original_jql != self.corrected_jql


def _is_correctable_value(v: str) -> bool:
    """Return False for values that should not be sent to the embedding store."""
    if not v:
        return False
    if _RE_FUNCTION.match(v):
        return False
    if _RE_DATE_EXPR.match(v):
        return False
    if _RE_ISSUE_KEY.match(v):
        return False
    return True


class JqlSemanticValidator:
    """Post-generation JQL field/value correction via embedding similarity."""

    def __init__(
        self,
        field_embeddings: "Jira_Field_Embeddings",
        value_embeddings: "JiraFieldValueEmbeddings | None",
        asset_embeddings: "JiraAssetEmbeddings | None",
        asset_field_ids:  set[str],
        model:            SentenceTransformer,
        field_threshold:  float = SEMANTIC_FIELD_THRESHOLD,
    ) -> None:
        self._field_emb       = field_embeddings
        self._value_emb       = value_embeddings
        self._asset_emb       = asset_embeddings
        self._asset_field_ids = asset_field_ids
        self._model           = model
        self._field_threshold = field_threshold

    async def validate(
        self,
        jql: str,
        clauses: list[LlmClause] | None = None,
    ) -> ValidatedJql:
        """Correct field names and values. Returns original JQL on any failure."""
        try:
            parsed = (
                self._clauses_from_llm(clauses, jql)
                if clauses
                else self._parse_clauses(jql)
            )

            validated: list[ValidatedClause] = []
            for clause in parsed:
                if not clause.correctable:
                    validated.append(ValidatedClause(
                        original=clause,
                        field_correction=None,
                        value_corrections=[],
                        rewritten_text=clause.raw_text,
                    ))
                    continue

                field_corr = self._correct_field(clause)
                val_corrs: list[SemanticValueCorrection] = []

                if field_corr and field_corr.type_compatible and clause.values:
                    is_asset = field_corr.field_id in self._asset_field_ids
                    for v in clause.values:
                        vc = await self._correct_value(field_corr.field_id, v, is_asset)
                        if vc:
                            val_corrs.append(vc)

                validated.append(ValidatedClause(
                    original=clause,
                    field_correction=field_corr,
                    value_corrections=val_corrs,
                    rewritten_text=self._rewrite_clause(clause, field_corr, val_corrs),
                ))

            corrected = self._rebuild_jql(jql, validated)
            return ValidatedJql(original_jql=jql, corrected_jql=corrected, clauses=validated)

        except Exception:
            logger.exception("SemanticValidator: unexpected error — returning original JQL")
            return ValidatedJql(original_jql=jql, corrected_jql=jql, clauses=[])

    # ------------------------------------------------------------------
    # Clause extraction
    # ------------------------------------------------------------------

    def _clauses_from_llm(
        self, clauses: list[LlmClause], jql: str
    ) -> list[JqlClause]:
        """Convert LlmClause list → JqlClause list. Skips non-correctable patterns."""
        result: list[JqlClause] = []
        for c in clauses:
            op = c.operator.strip()

            # aqlFunction clauses are omitted by the LLM per schema rules — skip any
            # that slipped through
            if _RE_AQL.search(str(c.value or "")):
                continue

            if isinstance(c.value, list):
                values = [v for v in c.value if _is_correctable_value(v)]
            elif c.value is None:
                values = []
            else:
                values = [c.value] if _is_correctable_value(c.value) else []

            # Find the substring in the JQL to use as raw_text for _rebuild_jql
            raw = self._locate_raw_text(c.field, op, c.value, jql)
            if raw is None:
                # Can't locate it safely — skip to avoid corrupting the JQL
                continue

            result.append(JqlClause(
                field_name=c.field,
                operator=op,
                values=values,
                raw_text=raw,
                correctable=True,
            ))
        return result

    def _parse_clauses(self, jql: str) -> list[JqlClause]:
        """Regex fallback for models that do not emit clauses[]."""
        result: list[JqlClause] = []
        seen_spans: list[tuple[int, int]] = []

        # Skip aqlFunction blocks entirely
        aql_spans = [m.span() for m in _RE_AQL.finditer(jql)]

        def _in_aql(start: int) -> bool:
            return any(a <= start for a, _ in aql_spans)

        # IN ('a', 'b', ...) — must check before single-value pattern
        for m in _RE_IN_CLAUSE.finditer(jql):
            if _in_aql(m.start()):
                continue
            op = re.sub(r"\s+", " ", m.group("op").upper())
            raw = m.group(0)
            vals_raw = m.group("vals")
            values = [
                v.strip().strip("'\"")
                for v in vals_raw.split(",")
                if _is_correctable_value(v.strip().strip("'\""))
            ]
            seen_spans.append(m.span())
            result.append(JqlClause(
                field_name=m.group("field").strip(),
                operator=op,
                values=values,
                raw_text=raw,
                correctable=True,
            ))

        # Single-value clauses
        for m in _RE_SINGLE_CLAUSE.finditer(jql):
            if _in_aql(m.start()):
                continue
            if any(s <= m.start() < e for s, e in seen_spans):
                continue
            op = re.sub(r"\s+", " ", m.group("op").upper())
            raw = m.group(0)
            val = m.group("sq") or m.group("dq") or ""
            values = [val] if _is_correctable_value(val) else []
            result.append(JqlClause(
                field_name=m.group("field").strip(),
                operator=op,
                values=values,
                raw_text=raw,
                correctable=True,
            ))

        return result

    def _locate_raw_text(
        self,
        field: str,
        op: str,
        value: str | list[str] | None,
        jql: str,
    ) -> str | None:
        """Find the clause substring in jql to use as the replacement anchor."""
        # Build a loose pattern from field + operator to locate the span
        escaped_field = re.escape(field)
        escaped_op    = re.escape(op).replace(r"\ ", r"\s+")
        pattern = re.compile(
            rf'(?i)"?{escaped_field}"?\s+{escaped_op}\s+'
            rf"""(?:'[^']*'|\([^)]*\)|[^\s,)]+)"""
        )
        m = pattern.search(jql)
        return m.group(0) if m else None

    # ------------------------------------------------------------------
    # Field correction (Step A + B)
    # ------------------------------------------------------------------

    def _correct_field(self, clause: JqlClause) -> FieldCorrection | None:
        result = self._field_emb.find_similar_field_name(
            name=clause.field_name,
            model=self._model,
            distance_threshold=self._field_threshold,
        )
        if result is None:
            return None

        field_id, display_name = result
        distance = self._lookup_distance(clause.field_name, field_id)
        field_type = self._lookup_field_type(field_id)
        compatible = self._type_compatible(clause.operator, field_type)

        if not compatible:
            logger.info(
                "SemanticValidator: type cross-check failed for %r → %r (type=%s, op=%s)",
                clause.field_name, display_name, field_type, clause.operator,
            )

        return FieldCorrection(
            original_name=clause.field_name,
            field_id=field_id,
            display_name=display_name,
            distance=distance,
            type_compatible=compatible,
        )

    def _lookup_field_type(self, field_id: str) -> str:
        """Fetch field_type for a known field_id from the embeddings table."""
        sql = f"SELECT field_type FROM {JIRA_FIELD_TABLE} WHERE field_id = %s LIMIT 1;"
        try:
            with PGVectorClient(self._field_emb.pgConfig) as pgclient:
                with pgclient.cursor() as cur:
                    cur.execute(sql, (field_id,))
                    row = cur.fetchone()
            return row[0] if row else ""
        except Exception:
            logger.debug("SemanticValidator: could not fetch field_type for %r", field_id)
            return ""

    def _lookup_distance(self, name: str, field_id: str) -> float:
        """Re-fetch the cosine distance for the matched field — used for logging."""
        try:
            emb = self._model.encode(name, normalize_embeddings=True)
            sql = (
                f"SELECT {JIRA_FIELD_COL_EMBEDDING} <-> %s::vector AS distance "
                f"FROM {JIRA_FIELD_TABLE} WHERE field_id = %s LIMIT 1;"
            )
            with PGVectorClient(self._field_emb.pgConfig) as pgclient:
                with pgclient.cursor() as cur:
                    cur.execute(sql, (emb.tolist(), field_id))
                    row = cur.fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _type_compatible(operator: str, field_type: str) -> bool:
        op = operator.strip().upper()
        if op in _NUMERIC_DATE_OPS:
            return not field_type or field_type.lower() in _NUMERIC_DATE_TYPES
        if op in _TEXT_OPS:
            return not field_type or field_type.lower() in _TEXT_TYPES
        return True

    # ------------------------------------------------------------------
    # Value correction (Step C)
    # ------------------------------------------------------------------

    async def _correct_value(
        self,
        field_id: str,
        value:    str,
        is_asset: bool,
    ) -> SemanticValueCorrection | None:
        emb = self._model.encode(value, normalize_embeddings=True)

        if not is_asset:
            if self._value_emb is None:
                return None
            hits = self._value_emb.find_similar_values_by_embedding(
                field_id=field_id,
                query_embedding=emb,
                top_n=1,
            )
            if hits and hits[0].distance < VALUE_AUTO_CORRECT_THRESHOLD:
                corrected = hits[0].value
                if corrected == value:
                    return None
                logger.info(
                    "SemanticValidator: value %r → %r (field=%s, dist=%.3f)",
                    value, corrected, field_id, hits[0].distance,
                )
                return SemanticValueCorrection(
                    original_value=value,
                    corrected_value=corrected,
                    distance=hits[0].distance,
                    source="standard",
                )
        else:
            if self._asset_emb is None:
                return None
            hits = self._asset_emb._search(field_id=field_id, embedding=emb, top_n=1)
            if hits and hits[0].distance < VALUE_AUTO_CORRECT_THRESHOLD:
                corrected = hits[0].object_name
                if corrected == value:
                    return None
                logger.info(
                    "SemanticValidator: asset value %r → %r (field=%s, dist=%.3f)",
                    value, corrected, field_id, hits[0].distance,
                )
                return SemanticValueCorrection(
                    original_value=value,
                    corrected_value=corrected,
                    distance=hits[0].distance,
                    source="asset",
                )
        return None

    # ------------------------------------------------------------------
    # JQL rewriting
    # ------------------------------------------------------------------

    def _rewrite_clause(
        self,
        clause:      JqlClause,
        field_corr:  FieldCorrection | None,
        val_corrs:   list[SemanticValueCorrection],
    ) -> str:
        """Build the replacement clause text from corrections."""
        if field_corr is None or not field_corr.type_compatible:
            return clause.raw_text

        display = field_corr.display_name
        # Quote multi-word field names
        field_str = f'"{display}"' if " " in display else display

        op = clause.operator

        # Asset field → rewrite as aqlFunction
        if field_corr.field_id in self._asset_field_ids and val_corrs:
            corrected_val = val_corrs[0].corrected_value
            return f'{field_str} IN aqlFunction(\'Name = "{corrected_val}"\')'

        # Build value map for substitution
        corrections_map = {vc.original_value: vc.corrected_value for vc in val_corrs}

        if not clause.values:
            # Numeric/function/IS comparison — field name only
            # Preserve original value text from raw_text after the operator
            op_pos = clause.raw_text.upper().find(op.upper())
            suffix = clause.raw_text[op_pos + len(op):] if op_pos >= 0 else ""
            return f"{field_str} {op}{suffix}"

        if op.upper() in ("IN", "NOT IN"):
            corrected_vals = [
                corrections_map.get(v, v) for v in clause.values
            ]
            vals_str = ", ".join(f"'{v}'" for v in corrected_vals)
            return f"{field_str} {op} ({vals_str})"

        corrected_val = corrections_map.get(clause.values[0], clause.values[0])
        return f"{field_str} {op} '{corrected_val}'"

    def _rebuild_jql(self, original_jql: str, validated: list[ValidatedClause]) -> str:
        """Substitute rewritten_text back into original_jql by raw_text match."""
        result = original_jql
        for vc in validated:
            raw = vc.original.raw_text
            rewritten = vc.rewritten_text
            if raw and rewritten and raw != rewritten and raw in result:
                result = result.replace(raw, rewritten, 1)
        return result
