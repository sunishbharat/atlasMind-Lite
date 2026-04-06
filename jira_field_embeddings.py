import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse
import psycopg2
from psycopg2.extras import Json as PgJson

import httpx
from sentence_transformers import SentenceTransformer
from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig
from sentence_transformers import SentenceTransformer
from settings import (
    DATABASE_URL, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    JIRA_FIELD_TABLE, JIRA_FIELD_COL_DESCRIPTION, JIRA_FIELD_COL_EMBEDDING, JIRA_FIELD_SEARCH_LIMIT,
    JIRA_FIELD_IGNORE_IDS,
)
from seed_manager import compute_file_hash, needs_reseeding, save_hash, get_stored_hash
from jira_field_api import fetch_and_save_fields, fetch_and_save_allowed_values
from settings import JIRA_ALLOWED_VALUES_FILENAME

import re
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


class Jira_Field_Embeddings:
    def __init__(
        self,
        config: EmbeddingsConfig = None,
        document_processor: DocumentProcessor = None,
    ):
        self.model_name = EMBEDDING_MODEL
        self.config = config if config else EmbeddingsConfig(model_name=self.model_name)

        # Reuse an existing DocumentProcessor (and its loaded SentenceTransformer)
        # when one is passed in — avoids loading the same model a second time.
        self.documentProc = document_processor if document_processor else DocumentProcessor(embedconfig=self.config)
        self.embedding_dim = self.documentProc._model.get_sentence_embedding_dimension()
        self.pgConfig = self.get_pgConfig_env()
        logger.info(f"Jira field Embedding dimension: {self.embedding_dim}")


    def run(self, jira_fields_file: Path):

        # Encode text documents into fixed-size vector embeddings using SentenceTransformer.
        self.setup_pgvector_db(self.pgConfig, self.embedding_dim)

        path = Path(jira_fields_file)
        if not path.exists():
            logger.info("%s not found — fetching from Jira REST API...", path.name)
            fetch_and_save_fields(path)

        av_file = path.parent / JIRA_ALLOWED_VALUES_FILENAME
        if not av_file.exists():
            logger.info("%s not found — fetching allowed values from Jira REST API...", av_file.name)
            asyncio.run(fetch_and_save_allowed_values(fields_json=path, output_json=av_file))

        self.seed_jira_field_embeddings_db(jira_fields_file)
        return self.documentProc._model

    def setup_pgvector_db(self, config: PGVectorConfig, embedding_dim: int = 384) -> None:
        """Create the pgvector extension and items table if they don't already exist.

        Args:
            config: PGVectorConfig with connection details.
            embedding_dim: Dimensionality of the embedding vectors (default 384 for all-MiniLM-L6-v2).
        """
        self._ensure_extension()
        with PGVectorClient(config) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {JIRA_FIELD_TABLE}(
                        id                    SERIAL PRIMARY KEY,
                        project_key           TEXT NOT NULL,        -- e.g. "ATLAS", "ENG"
                        field_id              TEXT NOT NULL,        -- e.g. "customfield_10023"
                        field_name            TEXT NOT NULL,        -- e.g. "Story Points"
                        field_type            TEXT,                 -- e.g. "number", "option", "user"
                        allowed_values        JSONB,                -- e.g. ["To Do", "In Progress", "Done"]
                        {JIRA_FIELD_COL_DESCRIPTION}  TEXT NOT NULL,
                        is_custom             BOOLEAN DEFAULT FALSE,
                        {JIRA_FIELD_COL_EMBEDDING}   vector({embedding_dim}),
                        created_at   TIMESTAMPTZ DEFAULT now()
                    );
                """)

    def _ensure_extension(self) -> None:
        conn = psycopg2.connect(
            database=self.pgConfig.database, user=self.pgConfig.user,
            password=self.pgConfig.password, host=self.pgConfig.host, port=self.pgConfig.port,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()


    def seed_jira_field_embeddings_db(self, jira_fields_file: Path) -> SentenceTransformer:
        """Parse a Jira fields JSON file and load embeddings into the pgvector database.

        Skips the encode + write cycle entirely if neither the fields file nor
        the allowed values file has changed since the last successful seed.
        A combined hash of both files is stored so that running
        fetch_and_save_allowed_values() automatically triggers a re-seed.

        Args:
            jira_fields_file: Path to the Jira fields JSON file from /rest/api/2/field.

        Returns:
            SentenceTransformer: The embedding model used, for reuse in similarity search.
        """
        jira_fields_file = Path(jira_fields_file)
        av_file = jira_fields_file.parent / JIRA_ALLOWED_VALUES_FILENAME

        # Combine hashes of both files so a change to either triggers re-seeding
        combined_hash = compute_file_hash(jira_fields_file)
        if av_file.exists():
            av_hash = compute_file_hash(av_file)
            combined_hash = hashlib.md5((combined_hash + av_hash).encode()).hexdigest()

        stored = get_stored_hash(self.pgConfig, str(jira_fields_file))
        if stored == combined_hash:
            logger.warning("%s unchanged — skipping re-seed.", jira_fields_file.name)
            return self.documentProc._model

        records = self._parse_jira_fields_json(str(jira_fields_file))
        logger.info("Loaded %d Jira field records from %s", len(records), jira_fields_file.name)
        self._update_pgvector_from_records(records, self.model_name)
        save_hash(self.pgConfig, str(jira_fields_file), combined_hash)
        return self.documentProc._model
    

    def get_pgConfig_env(self) -> PGVectorConfig:
        """Build a PGVectorConfig by parsing the DATABASE_URL environment variable.

        Returns:
            PGVectorConfig: Connection parameters (host, port, database, user, password)
            extracted from the DATABASE_URL connection string.
        """
        url = urlparse(DATABASE_URL)
        pgConfig = PGVectorConfig(
            database=url.path.lstrip("/"),   # jql_vectordb
            user=url.username,               # postgres
            password=url.password,           # postgres
            host=url.hostname,               # pgvector
            port=url.port,                   # 5432
        )
        return pgConfig


    async def search_jira_fields(
        self,
        query: str,
        model: SentenceTransformer,
        project_key: str = "",
    ) -> tuple[list[tuple], SentenceTransformer]:
        """Similarity search for Jira fields matching a natural language query.

        Encodes the query and retrieves the nearest field records from pgvector,
        ranked by embedding distance. An optional project_key narrows results
        to a specific Jira instance.

        model.encode is offloaded to a thread executor so it does not block
        the event loop. The DB call via PGVectorClient is synchronous — replace
        with an async DB client if full non-blocking behaviour is required.

        Args:
            query: Natural language description of the field to find,
                e.g. "who the ticket is assigned to" or "story points".
            model: SentenceTransformer model used to encode the query.
            project_key: When provided, restricts results to this project/instance.
                Omit to search across all loaded instances.

        Returns:
            tuple[list[tuple], SentenceTransformer]: A 2-tuple of:
                - list of (id, field_id, field_name, field_type, is_custom, description, distance) rows,
                  ordered by is_custom ASC then distance — system fields first, custom as fallback
                - the same model passed in, for chaining
        """
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(
            None,
            lambda: model.encode(query, normalize_embeddings=True),
        )

        # System fields (is_custom = false) are ranked before custom fields.
        # Distance is the tiebreaker within each group so the closest system
        # field always beats a custom field with the same semantic relevance.
        # A custom field only appears when no sufficiently close system field
        # fills the result set.
        if project_key:
            sql = f"""
                SELECT id, field_id, field_name, field_type, is_custom,
                       {JIRA_FIELD_COL_DESCRIPTION},
                       {JIRA_FIELD_COL_EMBEDDING} <-> %s AS distance
                FROM {JIRA_FIELD_TABLE}
                WHERE project_key = %s
                ORDER BY is_custom ASC, {JIRA_FIELD_COL_EMBEDDING} <-> %s
                LIMIT {JIRA_FIELD_SEARCH_LIMIT};
            """
            params = (query_emb, project_key, query_emb)
        else:
            sql = f"""
                SELECT id, field_id, field_name, field_type, is_custom,
                       {JIRA_FIELD_COL_DESCRIPTION},
                       {JIRA_FIELD_COL_EMBEDDING} <-> %s AS distance
                FROM {JIRA_FIELD_TABLE}
                ORDER BY is_custom ASC, {JIRA_FIELD_COL_EMBEDDING} <-> %s
                LIMIT {JIRA_FIELD_SEARCH_LIMIT};
            """
            params = (query_emb, query_emb)

        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        for id_, field_id, field_name, field_type, is_custom, description, dist in rows:
            logger.info(
                "field_id=%s  name=%r  type=%s  custom=%s  dist=%.4f",
                field_id, field_name, field_type, is_custom, dist,
            )

        return rows, model


    @staticmethod
    def _build_description(
        name: str,
        field_id: str,
        field_type: str,
        is_custom: bool,
        clause_names: list[str],
        allowed_values: list[str] | None = None,
    ) -> str:
        """Build a rich natural language description for embedding.

        Including allowed_values in the description lets the vector search map
        queries like "show me open tickets" directly to the Status field and its
        exact values, preventing the LLM from hallucinating values that do not
        exist in the project.

        Args:
            name: Human-readable field name, e.g. "Status".
            field_id: Field ID, e.g. "status" or "customfield_10023".
            field_type: Schema type string, e.g. "string", "option", "number".
            is_custom: True for custom fields, False for system fields.
            clause_names: JQL clause names from the field metadata.
            allowed_values: Discrete options for the field, e.g. ["To Do", "Done"].
                            When None the allowed values line is omitted.

        Returns:
            Formatted description string ready for embedding.
        """
        custom_label = "custom" if is_custom else "system"

        jql_clause = next(
            (c for c in clause_names if not c.startswith("cf[")),
            clause_names[0] if clause_names else field_id,
        )
        clause_str = (
            ", ".join(f"'{c}'" for c in clause_names)
            if len(clause_names) > 1
            else f"'{jql_clause}'"
        )

        parts = [f"{name}: a {custom_label} field of type {field_type}."]

        if allowed_values:
            parts.append(f"Allowed values: {', '.join(allowed_values)}.")

        parts.append(f"Used in JQL as {clause_str}.")
        parts.append(f"Field ID: {field_id}.")

        return " ".join(parts)


    def _parse_jira_fields_json(
        self,
        path: str,
        project_key: str = "",
        allowed_values_json: str = "",
    ) -> list[dict]:
        """Parse a Jira fields JSON file and build embeddable records.

        If an allowed values JSON file exists (produced by fetch_and_save_allowed_values
        in jira_field_api.py), it is loaded and merged so each eligible field gets
        a richer description that includes its discrete options. This lets the vector
        search map queries like "show me open tickets" to Status and its exact values.

        When no allowed values file is provided, the path is inferred by replacing
        the fields filename with jira_fields_allowed_values.json in the same directory.
        If that file does not exist the description is built without allowed values.

        Args:
            path: Filesystem path to the Jira fields JSON from /rest/api/2/field.
            project_key: Logical label for this Jira instance. Derived from the
                filename stem when omitted.
            allowed_values_json: Path to the allowed values JSON written by
                fetch_and_save_allowed_values(). Inferred from path when omitted.

        Returns:
            list[dict]: One dict per field with keys matching the jira_fields table schema.
        """
        import json as _json

        fields_path = Path(path)
        with open(fields_path, "r", encoding="utf-8") as f:
            raw: dict = _json.load(f)

        if not project_key:
            project_key = fields_path.stem

        # Load allowed values file — infer path if not provided
        av_path = Path(allowed_values_json) if allowed_values_json else \
            fields_path.parent / JIRA_ALLOWED_VALUES_FILENAME

        allowed_values_map: dict[str, list[str]] = {}
        if av_path.exists():
            allowed_values_map = _json.loads(av_path.read_text(encoding="utf-8"))
            logger.info("Loaded allowed values for %d fields from %s", len(allowed_values_map), av_path.name)
        else:
            logger.info("Allowed values file not found at %s — descriptions built without options", av_path)

        skipped = 0
        records: list[dict] = []
        for field_id, field in raw.items():
            name: str = field.get("name", field_id)
            schema: dict = field.get("schema") or {}
            field_type: str = schema.get("type", "unknown")
            is_custom: bool = bool(field.get("custom", False))
            clause_names: list[str] = field.get("clauseNames", [field_id])

            # Skip fields explicitly listed in the ignore set
            if field_id in JIRA_FIELD_IGNORE_IDS:
                skipped += 1
                continue

            # Skip custom fields with no human-readable clause name — only cf[...]
            # entries indicate the field has never been given a meaningful alias
            # and is unlikely to appear in user queries.
            if is_custom and all(c.startswith("cf[") for c in clause_names):
                skipped += 1
                continue
            allowed: list[str] | None = allowed_values_map.get(field_id) or None

            records.append({
                "project_key": project_key,
                "field_id": field_id,
                "field_name": name,
                "field_type": field_type,
                "allowed_values": allowed,
                "description": self._build_description(
                    name, field_id, field_type, is_custom, clause_names, allowed
                ),
                "is_custom": is_custom,
            })

        logger.info("Parsed %d Jira fields from %s (%d skipped)", len(records), fields_path.name, skipped)
        return records


    def _update_pgvector_from_records(
        self, records: list[dict], model_name: str
    ) -> SentenceTransformer:
        """Embed field descriptions and upsert all Jira field records into pgvector.

        Encodes the ``description`` of each record, then replaces all rows for
        the same ``project_key`` with the freshly embedded records. Using a
        scoped DELETE (by project_key) means records from other Jira instances
        are not affected when one instance is re-seeded.

        Args:
            records: Output of _parse_jira_fields_json() — list of dicts with
                keys: project_key, field_id, field_name, field_type,
                allowed_values, description, is_custom.
            model_name: SentenceTransformer model name (informational only;
                encoding uses self.documentProc._model).

        Returns:
            The SentenceTransformer model used for encoding.
        """
        if not records:
            logger.warning("_update_pgvector_from_records: empty records list, nothing to store.")
            return self.documentProc._model

        descriptions = [r["description"] for r in records]
        embeddings = self.documentProc._model.encode(
            descriptions,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("Encoding Jira field descriptions complete (%d vectors). Updating pgvector ...", len(embeddings))

        # All records in one file share the same project_key — delete that
        # project's existing rows before re-inserting to avoid duplicates.
        project_key = records[0]["project_key"]

        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {JIRA_FIELD_TABLE} WHERE project_key = %s;",
                    (project_key,),
                )
                for record, emb in zip(records, embeddings):
                    cur.execute(
                        f"""
                        INSERT INTO {JIRA_FIELD_TABLE}
                            (project_key, field_id, field_name, field_type,
                             allowed_values, {JIRA_FIELD_COL_DESCRIPTION}, is_custom, {JIRA_FIELD_COL_EMBEDDING})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            record["project_key"],
                            record["field_id"],
                            record["field_name"],
                            record["field_type"],
                            PgJson(record["allowed_values"]) if record["allowed_values"] is not None else None,
                            record["description"],
                            record["is_custom"],
                            emb.tolist(),
                        ),
                    )

        logger.info("pgvector updated with %d Jira field rows (project_key=%r).", len(records), project_key)
        return self.documentProc._model

