"""
Manages annotation-file hash tracking to avoid redundant re-seeding.

A lightweight `seed_metadata` table stores the MD5 hash of the last
successfully seeded annotation file. On each startup, needs_reseeding()
compares the current file hash against the stored one — the expensive
encode + DELETE/INSERT cycle only runs when the file has actually changed.
"""

import hashlib
import logging
from pathlib import Path

from pgvector_client import PGVectorClient, PGVectorConfig

logger = logging.getLogger(__name__)

_METADATA_TABLE = "seed_metadata"


def setup_metadata_table(config: PGVectorConfig) -> None:
    """Create the seed_metadata table if it doesn't already exist."""
    with PGVectorClient(config) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_METADATA_TABLE} (
                    source_file  TEXT PRIMARY KEY,
                    file_hash    TEXT NOT NULL,
                    seeded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)


def compute_file_hash(path: Path) -> str:
    """Return the MD5 hex digest of the file at *path*.

    The hash acts as a change-detector: encoding 400+ annotations through a
    SentenceTransformer is expensive, so we fingerprint the annotation file and
    only re-seed when the fingerprint differs from the one stored after the last
    successful seed. Identical file → identical hash → seed skipped entirely.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def get_stored_hash(config: PGVectorConfig, source_file: str) -> str | None:
    """Return the stored hash for *source_file*, or None if not yet seeded."""
    with PGVectorClient(config) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(
                f"SELECT file_hash FROM {_METADATA_TABLE} WHERE source_file = %s;",
                (source_file,),
            )
            row = cur.fetchone()
    return row[0] if row else None


def save_hash(config: PGVectorConfig, source_file: str, file_hash: str) -> None:
    """Upsert the hash for *source_file* after a successful seed."""
    with PGVectorClient(config) as pgclient:
        with pgclient.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_METADATA_TABLE} (source_file, file_hash, seeded_at)
                VALUES (%s, %s, now())
                ON CONFLICT (source_file)
                DO UPDATE SET file_hash = EXCLUDED.file_hash, seeded_at = now();
                """,
                (source_file, file_hash),
            )


def needs_reseeding(config: PGVectorConfig, annotation_file: Path) -> bool:
    """Return True if the annotation file has changed since the last seed.

    Sets up the metadata table on first call so callers don't need to.
    """
    setup_metadata_table(config)
    current_hash = compute_file_hash(annotation_file)
    stored_hash = get_stored_hash(config, str(annotation_file))
    if stored_hash == current_hash:
        logger.warning(f"{annotation_file} unchanged (hash {current_hash}) — skipping re-seed.")
        return False
    logger.warning(
        f"{annotation_file} changed (stored={stored_hash}, current={current_hash}) — re-seeding.",
    )
    return True
