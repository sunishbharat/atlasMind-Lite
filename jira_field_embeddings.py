import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse
import psycopg2

import httpx
from sentence_transformers import SentenceTransformer
from document_processor import DocumentProcessor
from dconfig import EmbeddingsConfig
from pgvector_client import PGVectorClient, PGVectorConfig
from sentence_transformers import SentenceTransformer
from settings import (
    DATABASE_URL, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    JIRA_FIELD_TABLE, JIRA_FIELD_COL_DESCRIPTION, JIRA_FIELD_COL_EMBEDDING, JIRA_FIELD_SEARCH_LIMIT,
)
from seed_manager import compute_file_hash, needs_reseeding, save_hash

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
            raise FileNotFoundError(f"Jira fields file not found: {path}")

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

        Skips the encode + write cycle entirely if the file has not changed
        since the last successful seed (hash-checked via seed_manager).

        Args:
            jira_fields_file: Path to the Jira fields JSON file from /rest/api/2/field.

        Returns:
            SentenceTransformer: The embedding model used, for reuse in similarity search.
        """
        jira_fields_file = Path(jira_fields_file)
        if not needs_reseeding(self.pgConfig, jira_fields_file):
            return self.documentProc._model

        records = self._parse_jira_fields_json(str(jira_fields_file))
        logger.info("Loaded %d Jira field records from %s", len(records), jira_fields_file.name)
        self._update_pgvector_from_records(records, self.model_name)
        save_hash(self.pgConfig, str(jira_fields_file), compute_file_hash(jira_fields_file))
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


   # async def generate_jql(self, query: str, model: SentenceTransformer) -> tuple[str, bool]:
   #     """Generate a JQL string (or general answer) from a natural language query using RAG + Ollama.

   #     Retrieves the top-5 most semantically similar (annotation, JQL) pairs from
   #     pgvector, builds a few-shot prompt, and sends it to the local Ollama LLM.

   #     The system prompt instructs Ollama to prefix JQL responses with ``<<JQL>>``.
   #     If the response starts with that tag the tag is stripped and ``is_general``
   #     is False.  Any other response is treated as a plain-text general answer and
   #     ``is_general`` is True — the caller should display it directly without
   #     hitting the Jira REST API.

   #     Args:
   #         query: The user's natural language query string.
   #         model: SentenceTransformer model used to encode the query for similarity search.

   #     Returns:
   #         tuple[str, bool]: ``(text, is_general)`` — *text* is either a raw JQL
   #         string or a plain-text answer; *is_general* is True for non-JQL responses.

   #     Raises:
   #         RuntimeError: If pgvector returns no examples (DB not loaded).
   #         httpx.HTTPStatusError: If the Ollama API request fails.
   #     """
   #     examples, _ = self.search_sample_jql_embeddings_db(query, model)
   #     if not examples:
   #         raise RuntimeError("No examples found in pgvector — was the DB loaded?")

   #     prompt = _build_prompt(query, examples)
   #     logger.debug("Ollama prompt:\n%s", prompt)

   #     async with httpx.AsyncClient(timeout=60) as client:
   #         response = await client.post(
   #             f"{OLLAMA_URL}/api/generate",
   #             json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": OLLAMA_TEMPERATURE}},
   #         )
   #         response.raise_for_status()
   #         text = response.json()["response"].strip()

   #     # Strip any accidental markdown fences
   #     text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE).strip("`").strip()

   #     if text.startswith(_JQL_TAG):
   #         return text[len(_JQL_TAG):].strip(), False

   #     logger.info("Non-JQL query detected — returning general answer without Jira REST API call.")
   #     logger.info("Response: %s", text)
   #     return text, True

    def _parse_jira_fields_json(self, path: str, project_key: str = "") -> list[dict]:
        """Parse a Jira fields JSON file (from /rest/api/2/field) into embeddable records.

        Builds a rich natural language description for each field so that user
        queries (e.g. "story points", "who it's assigned to") map well in
        semantic search. The description is what gets embedded and stored.

        Args:
            path: Filesystem path to the JSON file produced by the Jira REST API.
            project_key: Logical label for this Jira instance/project (e.g.
                "issues_apache_org"). Derived from the filename stem when omitted.

        Returns:
            list[dict]: One dict per field with keys matching the
            ``jira_fields`` table schema:
            ``project_key``, ``field_id``, ``field_name``, ``field_type``,
            ``allowed_values``, ``description``, ``is_custom``.
            ``allowed_values`` is always ``None`` — that data comes from a
            separate ``/field/{id}/option`` API call not available here.
        """
        import json as _json

        with open(path, "r", encoding="utf-8") as f:
            raw: dict = _json.load(f)

        if not project_key:
            project_key = Path(path).stem  # e.g. "jira_fields_issues_apache_org_jira"

        records: list[dict] = []
        for field_id, field in raw.items():
            name: str = field.get("name", field_id)
            schema: dict = field.get("schema") or {}  # guard against explicit null in JSON
            field_type: str = schema.get("type", "unknown")
            items_type: str = schema.get("items", "")          # e.g. "user" for array-of-user
            is_custom: bool = bool(field.get("custom", False))
            clause_names: list[str] = field.get("clauseNames", [field_id])

            # Prefer the human-readable clause name over the raw cf[...] form
            jql_clause: str = next(
                (c for c in clause_names if not c.startswith("cf[")),
                clause_names[0] if clause_names else field_id,
            )

            # Describe the type in plain English
            if items_type:
                type_label = f"array of {items_type}"
            else:
                type_label = field_type

            custom_label = "custom" if is_custom else "system"

            # If multiple clause names exist, list them all so queries using
            # either form can match (e.g. both "cf[12312322]" and the display name)
            if len(clause_names) > 1:
                clause_str = ", ".join(f"'{c}'" for c in clause_names)
            else:
                clause_str = f"'{jql_clause}'"

            description = (
                f"{name}: a {custom_label} field of type {type_label}. "
                f"Used in JQL as {clause_str}. "
                f"Field ID: {field_id}."
            )

            records.append({
                "project_key": project_key,
                "field_id": field_id,
                "field_name": name,
                "field_type": field_type,
                "allowed_values": None,   # populated separately via /field/{id}/option
                "description": description,
                "is_custom": is_custom,
            })

        logger.info("Parsed %d Jira fields from %s", len(records), Path(path).name)
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
                            record["allowed_values"],   # None → SQL NULL; pass psycopg2.extras.Json(...) when values exist
                            record["description"],
                            record["is_custom"],
                            emb.tolist(),
                        ),
                    )

        logger.info("pgvector updated with %d Jira field rows (project_key=%r).", len(records), project_key)
        return self.documentProc._model

