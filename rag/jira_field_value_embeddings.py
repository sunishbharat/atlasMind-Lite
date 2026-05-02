"""
jira_field_value_embeddings.py — Per-value vector store for allowed value correction.

Stores one embedding per (field_id, value) pair so JqlSanitizer can do
cosine similarity search to find the closest valid value for a hallucinated
one — no LLM call, no token cost.

Seeded from the allowed_values dict already loaded by
Jira_Field_Embeddings.fetch_allowed_values(). Hash-gated on the allowed
values file so a Jira change automatically triggers a re-seed.
"""

import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig
from rag.seed_manager import compute_file_hash, get_stored_hash, save_hash, setup_metadata_table
import numpy as np

from settings import (
    DATABASE_URL,
    EMBEDDING_BATCH_SIZE,
    JIRA_FIELD_VALUES_TABLE,
    JIRA_FIELD_VALUES_COL_EMBEDDING,
    MAX_VALUES_FOR_EMBEDDING,
    VALUE_HINT_MAX_CANDIDATES,
    VALUE_PROMPT_MAX_CANDIDATES,
)

logger = logging.getLogger(__name__)

# Seed key suffix distinguishes this table's hash from the main field embeddings hash
# stored for the same source file.
_SEED_KEY_SUFFIX = "::field_values"


class FieldValueRecord(BaseModel):
    """A single allowed value for a Jira field, ready for embedding."""
    field_id: str
    field_name: str
    value: str


class SimilarValue(BaseModel):
    """A candidate value returned by cosine similarity search."""
    value: str
    distance: float


class JiraFieldValueEmbeddings:
    """Manages the jira_field_values vector table.

    One row per (field_id, value) pair. Enables cosine similarity search
    to find the closest valid value for a hallucinated one without any LLM call.

    Usage::

        fve = JiraFieldValueEmbeddings(embedconfig, document_processor)
        fve.setup_table()
        fve.seed(allowed_values, id_to_name, av_file)

        candidates = fve.find_similar_values("status", "closed", model, top_n=3)
        # → [SimilarValue(value="Done", distance=0.12), ...]
    """

    def __init__(
        self,
        config: EmbeddingsConfig,
        document_processor: DocumentProcessor,
    ) -> None:
        self.pgConfig = self._build_pg_config()
        self.documentProc = document_processor

    def _build_pg_config(self) -> PGVectorConfig:
        url = urlparse(DATABASE_URL)
        return PGVectorConfig(
            database=url.path.lstrip("/"),
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port,
        )

    def setup_table(self) -> None:
        """Create the jira_field_values table and field_id index if they do not exist."""
        dim = self.documentProc._model.get_sentence_embedding_dimension()
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {JIRA_FIELD_VALUES_TABLE} (
                        id          SERIAL PRIMARY KEY,
                        field_id    TEXT NOT NULL,
                        field_name  TEXT NOT NULL,
                        value       TEXT NOT NULL,
                        {JIRA_FIELD_VALUES_COL_EMBEDDING} vector({dim}),
                        created_at  TIMESTAMPTZ DEFAULT now()
                    );
                    CREATE INDEX IF NOT EXISTS idx_{JIRA_FIELD_VALUES_TABLE}_field_id
                        ON {JIRA_FIELD_VALUES_TABLE}(field_id);
                """)
        logger.info("jira_field_values table ready (dim=%d).", dim)

    def seed(
        self,
        allowed_values: dict[str, list[str]],
        id_to_name: dict[str, str],
        source_file: Path,
    ) -> None:
        """Embed and store all allowed values, skipping if unchanged.

        Hash-gated on source_file: if the allowed values file has not changed
        since the last successful seed the encode + DELETE/INSERT cycle is skipped.

        Args:
            allowed_values: {field_id: [value_strings]} from fetch_allowed_values().
            id_to_name:     {field_id: display_name} for populating field_name column.
            source_file:    Path to the allowed values JSON file (hash-gating key).
        """
        setup_metadata_table(self.pgConfig)
        seed_key = str(source_file) + _SEED_KEY_SUFFIX + f"::cap{MAX_VALUES_FOR_EMBEDDING}"
        current_hash = compute_file_hash(source_file)
        stored = get_stored_hash(self.pgConfig, seed_key)
        if stored == current_hash:
            logger.info("jira_field_values: allowed values unchanged — skipping re-seed.")
            return

        records = self._build_records(allowed_values, id_to_name)
        if not records:
            logger.warning("jira_field_values: no records to seed — allowed_values is empty.")
            return

        values_text = [r.value for r in records]
        embeddings = self.documentProc._model.encode(
            values_text,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("jira_field_values: encoded %d value embeddings.", len(embeddings))

        rows = [
            (r.field_id, r.field_name, r.value, emb.tolist())
            for r, emb in zip(records, embeddings)
        ]
        _BATCH = 500
        logger.info(
            "jira_field_values: inserting %d rows in batches of %d...",
            len(rows), _BATCH,
        )
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {JIRA_FIELD_VALUES_TABLE};")
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {JIRA_FIELD_VALUES_TABLE}
                        (field_id, field_name, value, {JIRA_FIELD_VALUES_COL_EMBEDDING})
                    VALUES %s
                    """,
                    rows,
                    page_size=_BATCH,
                )

        save_hash(self.pgConfig, seed_key, current_hash)
        logger.info(
            "jira_field_values: seeded %d value embeddings across %d fields.",
            len(records),
            len(allowed_values),
        )

    def find_similar_values(
        self,
        field_id: str,
        bad_value: str,
        model: SentenceTransformer,
        top_n: int = VALUE_HINT_MAX_CANDIDATES,
    ) -> list[SimilarValue]:
        """Find the closest valid values for bad_value within field_id.

        Encodes bad_value with the same model used at seeding time and runs
        a cosine similarity search restricted to rows for field_id.
        No LLM call is made — this is a pure pgvector DB query.

        Args:
            field_id:  Jira field ID to restrict the search to (e.g. "status").
            bad_value: Hallucinated or misspelled value from LLM-produced JQL.
            model:     SentenceTransformer — must be the same model used at seeding.
            top_n:     Maximum number of candidates to return.

        Returns:
            List of SimilarValue ordered by distance ascending (closest first).
            Empty when the table has no rows for field_id.
        """
        embedding = model.encode(bad_value, normalize_embeddings=True)
        sql = f"""
            SELECT value, {JIRA_FIELD_VALUES_COL_EMBEDDING} <-> %s::vector AS distance
            FROM {JIRA_FIELD_VALUES_TABLE}
            WHERE field_id = %s
            ORDER BY distance
            LIMIT %s;
        """
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(sql, (embedding.tolist(), field_id, top_n))
                rows = cur.fetchall()

        results = [SimilarValue(value=row[0], distance=row[1]) for row in rows]
        if results:
            logger.debug(
                "jira_field_values: similarity search for %r in field %r → %s",
                bad_value, field_id,
                [(r.value, round(r.distance, 3)) for r in results],
            )
        return results

    def find_similar_values_by_embedding(
        self,
        field_id: str,
        query_embedding: np.ndarray,
        top_n: int = VALUE_PROMPT_MAX_CANDIDATES,
    ) -> list[SimilarValue]:
        """Find closest valid values using a pre-computed query embedding.

        Identical to find_similar_values() but takes an already-encoded vector
        instead of a raw string. Use this in _build_prompt() where the same
        query embedding is reused across multiple field searches — avoids
        re-encoding the query string for every field.

        Args:
            field_id:        Jira field ID to restrict the search to.
            query_embedding: Normalised embedding vector for the user query.
            top_n:           Maximum number of candidates to return.

        Returns:
            List of SimilarValue ordered by distance ascending (closest first).
        """
        sql = f"""
            SELECT value, {JIRA_FIELD_VALUES_COL_EMBEDDING} <-> %s::vector AS distance
            FROM {JIRA_FIELD_VALUES_TABLE}
            WHERE field_id = %s
            ORDER BY distance
            LIMIT %s;
        """
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(sql, (query_embedding.tolist(), field_id, top_n))
                rows = cur.fetchall()

        results = [SimilarValue(value=row[0], distance=row[1]) for row in rows]
        if results:
            logger.debug(
                "jira_field_values: prompt-hint search for field %r → %s",
                field_id,
                [(r.value, round(r.distance, 3)) for r in results],
            )
        return results

    @staticmethod
    def _build_records(
        allowed_values: dict[str, list[str]],
        id_to_name: dict[str, str],
    ) -> list[FieldValueRecord]:
        """Flatten allowed_values into one FieldValueRecord per (field_id, value) pair.

        High-cardinality fields (versions, components, labels) are capped at
        MAX_VALUES_FOR_EMBEDDING. Exact-match correction is unaffected — the
        sanitizer's in-memory _normed dict holds all values regardless of this cap.
        """
        records: list[FieldValueRecord] = []
        skipped_fields = 0
        for field_id, values in allowed_values.items():
            field_name = id_to_name.get(field_id, field_id)
            capped = values[:MAX_VALUES_FOR_EMBEDDING]
            if len(values) > MAX_VALUES_FOR_EMBEDDING:
                skipped_fields += 1
            for value in capped:
                records.append(FieldValueRecord(
                    field_id=field_id,
                    field_name=field_name,
                    value=value,
                ))
        if skipped_fields:
            logger.info(
                "jira_field_values: capped %d high-cardinality field(s) to %d values each "
                "(exact-match correction unaffected — in-memory dict holds all values).",
                skipped_fields, MAX_VALUES_FOR_EMBEDDING,
            )
        return records
