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
    JQL_TABLE, JQL_COL_ANNOTATION, JQL_COL_JQL, JQL_COL_EMBEDDING, JQL_SEARCH_LIMIT,
)
from rag.seed_manager import compute_file_hash, needs_reseeding, save_hash

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


class JQL_Embeddings:
    def __init__(self, config: EmbeddingsConfig=None, documentProc: DocumentProcessor=None):
        self.model_name = EMBEDDING_MODEL
        self.config = config if config else EmbeddingsConfig(model_name=self.model_name)

        self.documentProc = documentProc if documentProc else DocumentProcessor(embedconfig=self.config)
        self.embedding_dim = self.documentProc._model.get_sentence_embedding_dimension()
        self.pgConfig = self.get_pgConfig_env()
        logger.info(f"Embedding dimension: {self.embedding_dim}")


    def run(self, annotation_file: Path):

        # Encode text documents into fixed-size vector embeddings using SentenceTransformer.
        self.setup_pgvector_db(self.pgConfig, self.embedding_dim)

        path = Path(annotation_file)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        self.seed_sample_jql_embeddings_db(annotation_file)
        return self.documentProc, self.documentProc._model

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
                    CREATE TABLE IF NOT EXISTS {JQL_TABLE}(
                        id                    SERIAL PRIMARY KEY,
                        {JQL_COL_ANNOTATION}  TEXT NOT NULL,
                        {JQL_COL_JQL}         TEXT NOT NULL,
                        {JQL_COL_EMBEDDING}   vector({embedding_dim})
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


    def seed_sample_jql_embeddings_db(self, annotation_file: Path) -> SentenceTransformer:
        """Parse a JQL annotation file and load embeddings into the pgvector database.

        Skips the encode + write cycle entirely if the annotation file has not
        changed since the last successful seed (hash-checked via seed_manager).

        Args:
            annotation_file: Path to the annotation file (.md format with comment/JQL pairs).

        Returns:
            SentenceTransformer: The embedding model used, for reuse in similarity search.
        """
        annotation_file = Path(annotation_file)
        if not needs_reseeding(self.pgConfig, annotation_file):
            return self.documentProc._model

        pairs = self._parse_jql_annotations(str(annotation_file))
        logger.info("Loaded %d annotation pairs from %s", len(pairs), annotation_file.name)
        self._update_pgvector_from_annotations(pairs, self.model_name)
        save_hash(self.pgConfig, str(annotation_file), compute_file_hash(annotation_file))
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


    def search_sample_jql_embeddings_db(self, query: str, model: SentenceTransformer) -> tuple[list[tuple], SentenceTransformer]:
        """Similarity search against the annotation/jql schema in pgvector.

        Encodes the query and retrieves the top-5 nearest neighbours from the
        `items` table, selecting the `annotation` and `jql` columns. Used by
        generate_jql() in main.py to build the few-shot RAG prompt.

        Args:
            query: Natural language query string to encode and search.
            model: SentenceTransformer model used to encode the query.

        Returns:
            tuple[list[tuple], SentenceTransformer]: A 2-tuple of:
                - list of (id, annotation, jql, distance) rows from pgvector
                - the same model passed in (for chaining)
        """
        query_emb = model.encode(query, normalize_embeddings=True)

        sql = f"""
            SELECT id, {JQL_COL_ANNOTATION}, {JQL_COL_JQL}, {JQL_COL_EMBEDDING} <-> %s AS distance
            FROM {JQL_TABLE}
            ORDER BY {JQL_COL_EMBEDDING} <-> %s
            LIMIT {JQL_SEARCH_LIMIT};
            """

        logging.info(f"SQL Query = {sql}")

        with PGVectorClient(self.pgConfig) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(sql, (query_emb, query_emb))
                rows = cur.fetchall()

        for id_, annotation, jql_text, dist in rows:
            logging.info("*" * 40)
            logging.info(f"{id_=}, {dist=}, ->\n annotation: {annotation}\n jql: {jql_text}")

        return rows, model


    def _parse_jql_annotations(self, path: str) -> list[dict[str, str]]:
        """Parse a JQL annotation file and return comment/JQL pairs.

        Reads the file at *path* and extracts all block-comment-style annotation
        pairs in the format ``/* comment */\\nJQL``. Each pair becomes a dict with
        keys ``"comment"`` and ``"jql"``.

        Args:
            path: Filesystem path to the annotation file (UTF-8 encoded Markdown).

        Returns:
            list[dict[str, str]]: Ordered list of ``{"comment": ..., "jql": ...}``
            dicts, one per matched annotation block. Returns an empty list if no
            pairs are found.
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info("File loaded: %d characters", len(text))

        pattern = re.compile(r'/\*\s*(.*?)\s*\*/[ \t]*[\r\n]+(?!/\*)([^\r\n]+)', re.DOTALL)
        pairs: list[dict[str, str]] = [
            {"comment": m.group(1).strip(), "jql": m.group(2).strip()}
            for m in pattern.finditer(text)
        ]

        logger.info("Parsed %d comment/JQL pairs", len(pairs))

        for p in pairs[-10:]:
            logger.debug("comment: %s", p["comment"])
            logger.debug("jql:     %s", p["jql"])

        return pairs


    def _update_pgvector_from_annotations(
        self, pairs: list[dict[str, str]],
        model_name: str) -> SentenceTransformer:
    
        """Embed annotation comments and upsert (comment, jql, embedding) rows into pgvector.

        Args:
            pairs: Output of _parse_jql_annotations() — list of {"comment": ..., "jql": ...} dicts.
            model_name: SentenceTransformer model used for encoding. Defaults to module-level model_1.

        Returns:
            The SentenceTransformer model used for encoding (reuse for similarity search).
        """
        if not pairs:
            logging.warning("update_pgvector_from_annotations: empty pairs list, nothing to store.")
            return

        comments = [p["comment"] for p in pairs]
        embeddings = self.documentProc._model.encode(comments, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)

        logging.info("Encoding complete (%d vectors). Updating pgvector ...", len(embeddings))
        with PGVectorClient(self.get_pgConfig_env()) as pgclient:
            with pgclient.cursor() as cur:
                cur.execute(f"DELETE FROM {JQL_TABLE};")
                for pair, emb in zip(pairs, embeddings):
                    cur.execute(
                        f"INSERT INTO {JQL_TABLE} ({JQL_COL_ANNOTATION}, {JQL_COL_JQL}, {JQL_COL_EMBEDDING}) VALUES (%s, %s, %s);",
                        (pair["comment"], pair["jql"], emb.tolist()),
                    )
        logging.info("pgvector updated with %d rows.", len(pairs))
        return self.documentProc._model
        