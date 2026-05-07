"""
jira_asset_embeddings.py — Per-label vector store for Jira Assets values.

Stores one embedding per (field_id, label) pair from jira_assets.json so
_build_prompt can inject query-ranked asset object labels as value hints
before LLM generation — reducing hallucinated or partial asset labels in
generated JQL (e.g. "Sample Object" → correct aqlFunction form).

Two columns are stored per row:
  label       — full label as returned by the Assets AQL API, e.g.
                "Sample Object (ABCD-1234)". Embedded for cosine search.
  object_name — clean name with (KEY-NNN) suffix stripped, e.g.
                "Sample Object". Used in generated JQL:
                aqlFunction('Name = "Sample Object"')

Seeded from data/<hostname>/jira_assets.json. Hash-gated so re-running
refresh_asset_values() automatically triggers re-seeding on next startup.

Asset fields are intentionally kept in a separate table from jira_field_values
because they come from a different API, have a different object model (label vs
option value), and may need independent refresh cycles.
"""

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from psycopg2.extras import execute_values
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig
from rag.seed_manager import compute_file_hash, get_stored_hash, save_hash, setup_metadata_table
from settings import (
    DATABASE_URL,
    EMBEDDING_BATCH_SIZE,
    JIRA_ASSET_VALUES_COL_EMBEDDING,
    JIRA_ASSET_VALUES_TABLE,
    VALUE_HINT_MAX_CANDIDATES,
    VALUE_PROMPT_MAX_CANDIDATES,
)

logger = logging.getLogger(__name__)

_SEED_KEY_SUFFIX = "::asset_values"

# Strips the Assets object key suffix from labels, e.g. " (ABCD-1234)" → "".
_KEY_SUFFIX_RE = re.compile(r"\s+\([A-Z]+-\d+\)\s*$")


def _strip_key_suffix(label: str) -> str:
    """Return the label with the Assets object key suffix removed.

    "Sample Object (ABCD-1234)"  → "Sample Object"
    "Another Item (XY-999)"      → "Another Item"
    "Plain Label"                → "Plain Label"
    """
    return _KEY_SUFFIX_RE.sub("", label).strip()


class AssetValueRecord(BaseModel):
    field_id: str
    field_name: str
    object_type: str
    label: str
    object_name: str


class SimilarAssetValue(BaseModel):
    label: str
    object_name: str
    distance: float


class JiraAssetEmbeddings:
    """Manages the jira_asset_values pgvector table.

    One row per (field_id, label) pair. Enables cosine similarity search to
    find the closest valid asset object label for a partial or misspelled value
    in user queries — without any LLM call.

    Usage::

        ae = JiraAssetEmbeddings(embedconfig, document_processor)
        ae.setup_table()
        ae.seed(Path("data/myorg/jira_assets.json"))

        candidates = ae.find_similar_values("customfield_10200", "driving", model)
        # → [SimilarAssetValue(label="Sample Object (ABCD-1234)",
        #                      object_name="Sample Object", distance=0.08), ...]
    """

    def __init__(self, config: EmbeddingsConfig, document_processor: DocumentProcessor) -> None:
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
        """Create the jira_asset_values table and field_id index if they do not exist.

        Also adds the object_name column if upgrading from a prior schema that lacked it.
        """
        dim = self.documentProc._model.get_embedding_dimension()
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {JIRA_ASSET_VALUES_TABLE} (
                        id          SERIAL PRIMARY KEY,
                        field_id    TEXT NOT NULL,
                        field_name  TEXT NOT NULL,
                        object_type TEXT NOT NULL,
                        label       TEXT NOT NULL,
                        object_name TEXT NOT NULL DEFAULT '',
                        {JIRA_ASSET_VALUES_COL_EMBEDDING} vector({dim}),
                        created_at  TIMESTAMPTZ DEFAULT now()
                    );
                    CREATE INDEX IF NOT EXISTS idx_{JIRA_ASSET_VALUES_TABLE}_field_id
                        ON {JIRA_ASSET_VALUES_TABLE}(field_id);
                """)
                cur.execute(f"""
                    ALTER TABLE {JIRA_ASSET_VALUES_TABLE}
                        ADD COLUMN IF NOT EXISTS object_name TEXT NOT NULL DEFAULT '';
                """)
        logger.info("jira_asset_values table ready (dim=%d).", dim)

    def seed(self, source_file: Path) -> None:
        """Embed and store all asset labels from source_file, skipping if unchanged.

        Hash-gated on source_file: if jira_assets.json has not changed since the
        last successful seed the encode + TRUNCATE/INSERT cycle is skipped.

        Args:
            source_file: Path to data/<hostname>/jira_assets.json.
        """
        if not source_file.exists():
            logger.info(
                "jira_asset_values: %s not found — run refresh_asset_values() first.",
                source_file,
            )
            return

        setup_metadata_table(self.pgConfig)
        seed_key = str(source_file) + _SEED_KEY_SUFFIX
        current_hash = compute_file_hash(source_file)
        if get_stored_hash(self.pgConfig, seed_key) == current_hash:
            logger.info("jira_asset_values: asset values unchanged — skipping re-seed.")
            return

        data: dict = json.loads(source_file.read_text(encoding="utf-8"))
        records: list[AssetValueRecord] = [
            AssetValueRecord(
                field_id=field_id,
                field_name=cfg["display_name"],
                object_type=cfg["object_type"],
                label=label,
                object_name=_strip_key_suffix(label),
            )
            for field_id, cfg in data.items()
            for label in cfg.get("labels", [])
        ]

        if not records:
            logger.warning("jira_asset_values: no labels to seed — jira_assets.json is empty.")
            return

        embeddings = self.documentProc._model.encode(
            [r.label for r in records],
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        rows = [
            (r.field_id, r.field_name, r.object_type, r.label, r.object_name, emb.tolist())
            for r, emb in zip(records, embeddings)
        ]
        _BATCH = 500
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {JIRA_ASSET_VALUES_TABLE};")
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {JIRA_ASSET_VALUES_TABLE}
                        (field_id, field_name, object_type, label, object_name,
                         {JIRA_ASSET_VALUES_COL_EMBEDDING})
                    VALUES %s
                    """,
                    rows,
                    page_size=_BATCH,
                )

        save_hash(self.pgConfig, seed_key, current_hash)
        logger.info(
            "jira_asset_values: seeded %d labels across %d field(s).",
            len(records),
            len(data),
        )

    def find_similar_values(
        self,
        field_id: str,
        query: str,
        model: SentenceTransformer,
        top_n: int = VALUE_HINT_MAX_CANDIDATES,
    ) -> list[SimilarAssetValue]:
        """Find the closest asset labels for a query string within field_id.

        Encodes query with the same model used at seeding time and runs a
        cosine similarity search restricted to rows for field_id.

        Args:
            field_id: Jira custom field ID, e.g. "customfield_10200".
            query:    Raw search term, e.g. "driving functions".
            model:    SentenceTransformer — must be the same model used at seeding.
            top_n:    Maximum candidates to return.

        Returns:
            List of SimilarAssetValue ordered by distance ascending (closest first).
        """
        embedding = model.encode(query, normalize_embeddings=True)
        return self._search(field_id, embedding, top_n)

    def find_similar_values_by_embedding(
        self,
        field_id: str,
        query_embedding: np.ndarray,
        top_n: int = VALUE_PROMPT_MAX_CANDIDATES,
    ) -> list[SimilarAssetValue]:
        """Find closest asset labels using a pre-computed query embedding.

        Use this in _build_prompt() where the same query embedding is reused
        across multiple field searches — avoids re-encoding the query string.

        Args:
            field_id:        Jira custom field ID.
            query_embedding: Normalised embedding vector for the user query.
            top_n:           Maximum candidates to return.

        Returns:
            List of SimilarAssetValue ordered by distance ascending (closest first).
        """
        return self._search(field_id, query_embedding, top_n)

    def _search(
        self,
        field_id: str,
        embedding: np.ndarray,
        top_n: int,
    ) -> list[SimilarAssetValue]:
        sql = f"""
            SELECT label, object_name, {JIRA_ASSET_VALUES_COL_EMBEDDING} <-> %s::vector AS distance
            FROM {JIRA_ASSET_VALUES_TABLE}
            WHERE field_id = %s
            ORDER BY distance
            LIMIT %s;
        """
        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(sql, (embedding.tolist(), field_id, top_n))
                rows = cur.fetchall()

        results = [
            SimilarAssetValue(label=row[0], object_name=row[1], distance=row[2])
            for row in rows
        ]
        if results:
            logger.debug(
                "jira_asset_values: search for field %r → %s",
                field_id,
                [(r.object_name, round(r.distance, 3)) for r in results],
            )
        return results
