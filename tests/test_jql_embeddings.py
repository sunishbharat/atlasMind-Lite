import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from pgvector_client import PGVectorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pgclient_mock(mock_cur):
    """Return a PGVectorClient context-manager mock wired to mock_cur."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


def _make_jql_instance(mock_model=None, mock_cur=None):
    """Shared patch context for constructing JQL_Embeddings without real DB/model."""
    mock_cur = mock_cur or MagicMock()
    mock_model = mock_model or MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_model.encode.return_value = np.array([[0.1] * 768, [0.4] * 768])
    mock_processor = MagicMock()
    mock_processor._model = mock_model
    return mock_cur, mock_model, mock_processor


# ---------------------------------------------------------------------------
# Fixture: __init__ now takes no annotation_file — only config.
# Mocks only DB + embedding I/O; real Python logic runs.
# ---------------------------------------------------------------------------

@pytest.fixture
def jql():
    mock_cur, mock_model, mock_processor = _make_jql_instance()

    with patch("jql_embeddings.EmbeddingsConfig"), \
         patch("jql_embeddings.DocumentProcessor", return_value=mock_processor), \
         patch("jql_embeddings.EMBEDDING_MODEL", "all-MiniLM-L6-v2"), \
         patch("jql_embeddings.JQL_Embeddings.get_pgConfig_env", return_value=MagicMock()):
        from jql_embeddings import JQL_Embeddings
        instance = JQL_Embeddings()

    instance.model = mock_model
    instance._mock_cur = mock_cur
    return instance


# ---------------------------------------------------------------------------
# __init__: embedding dim is derived from model before setup_pgvector_db
# ---------------------------------------------------------------------------

def test_run_passes_correct_embedding_dim_to_setup(jql, tmp_path):
    """Regression: run() must pass the model's actual embedding dimension to
    setup_pgvector_db — hardcoded 384 caused psycopg2.errors.DataException at insert time."""
    annotation_file = tmp_path / "annotations.md"
    annotation_file.write_text("/* test */\nproject = X\n")
    jql.documentProc._model.get_sentence_embedding_dimension.return_value = 768

    seen_dim = {}

    def capture_setup(self, config, embedding_dim=384):
        seen_dim["value"] = embedding_dim

    with patch("jql_embeddings.JQL_Embeddings.setup_pgvector_db", capture_setup), \
         patch.object(jql, "seed_sample_jql_embeddings_db"):
        jql.run(annotation_file)

    assert seen_dim.get("value") == 768, \
        f"setup_pgvector_db called with dim={seen_dim.get('value')}, expected 768"


# ---------------------------------------------------------------------------
# get_pgConfig_env
# ---------------------------------------------------------------------------

def test_get_pgConfig_env(jql):
    url = "postgresql://postgres:secret@pgvector:5432/jql_vectordb"
    with patch("jql_embeddings.DATABASE_URL", url):
        from jql_embeddings import JQL_Embeddings
        config = JQL_Embeddings.get_pgConfig_env(jql)

    assert isinstance(config, PGVectorConfig)
    assert config.host == "pgvector"
    assert config.port == 5432
    assert config.database == "jql_vectordb"
    assert config.user == "postgres"
    assert config.password == "secret"


# ---------------------------------------------------------------------------
# setup_pgvector_db
# ---------------------------------------------------------------------------

def test_setup_pgvector_db_executes_ddl(jql):
    mock_cur = MagicMock()
    config = PGVectorConfig(host="localhost", port=5432, database="db", user="u", password="p")
    mock_psycopg2_conn = MagicMock()
    mock_psycopg2_conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_psycopg2_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("jql_embeddings.PGVectorClient", return_value=_make_pgclient_mock(mock_cur)), \
         patch("jql_embeddings.psycopg2.connect", return_value=mock_psycopg2_conn):
        from jql_embeddings import JQL_Embeddings
        JQL_Embeddings.setup_pgvector_db(jql, config)

    executed = [str(c) for c in mock_cur.execute.call_args_list]
    assert any("CREATE EXTENSION" in s for s in executed)
    assert any("CREATE TABLE" in s for s in executed)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_raises_when_file_missing(jql):
    with pytest.raises(FileNotFoundError, match="Annotation file not found"):
        with patch.object(jql, "setup_pgvector_db"):
            jql.run("/nonexistent/annotations.md")


def test_run_calls_seed_and_returns_model(jql, tmp_path):
    annotation_file = tmp_path / "annotations.md"
    annotation_file.write_text("/* open bugs */\nproject = TEST\n")

    with patch.object(jql, "setup_pgvector_db"), \
         patch.object(jql, "seed_sample_jql_embeddings_db") as mock_seed:
        result = jql.run(annotation_file)

    mock_seed.assert_called_once_with(annotation_file)
    assert result is jql.documentProc._model


# ---------------------------------------------------------------------------
# seed_sample_jql_embeddings_db — skips when file unchanged
# ---------------------------------------------------------------------------

def test_seed_skips_when_file_unchanged(jql, tmp_path):
    annotation_file = tmp_path / "annotations.md"
    annotation_file.write_text("/* open bugs */\nproject = TEST\n")

    with patch("jql_embeddings.needs_reseeding", return_value=False) as mock_check, \
         patch.object(jql, "_update_pgvector_from_annotations") as mock_update:
        jql.seed_sample_jql_embeddings_db(annotation_file)

    mock_check.assert_called_once_with(jql.pgConfig, annotation_file)
    mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# seed_sample_jql_embeddings_db — re-seeds and saves hash when file changed
# ---------------------------------------------------------------------------

def test_seed_reseeds_and_saves_hash_when_file_changed(jql, tmp_path):
    annotation_file = tmp_path / "annotations.md"
    annotation_file.write_text("/* open bugs */\nproject = TEST AND status = Open\n")

    with patch("jql_embeddings.needs_reseeding", return_value=True), \
         patch.object(jql, "_update_pgvector_from_annotations") as mock_update, \
         patch("jql_embeddings.save_hash") as mock_save, \
         patch("jql_embeddings.compute_file_hash", return_value="abc123") as mock_hash:
        jql.seed_sample_jql_embeddings_db(annotation_file)

    mock_update.assert_called_once()
    mock_save.assert_called_once_with(jql.pgConfig, str(annotation_file), "abc123")


# ---------------------------------------------------------------------------
# _update_pgvector_from_annotations — inserts correct SQL
# ---------------------------------------------------------------------------

def test_update_pgvector_inserts_rows(jql):
    pairs = [
        {"comment": "open bugs", "jql": "project = TEST AND status = Open"},
        {"comment": "my tickets", "jql": "assignee = currentUser()"},
    ]
    fake_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    jql.documentProc._model.encode.return_value = fake_embeddings

    mock_cur = MagicMock()
    with patch("jql_embeddings.PGVectorClient", return_value=_make_pgclient_mock(mock_cur)), \
         patch("jql_embeddings.EMBEDDING_BATCH_SIZE", 32, create=True), \
         patch.object(jql, "get_pgConfig_env", return_value=MagicMock()):
        jql._update_pgvector_from_annotations(pairs, model_name="all-MiniLM-L6-v2")

    executed_sql = [c.args[0] for c in mock_cur.execute.call_args_list]
    assert any("DELETE FROM" in s for s in executed_sql)
    insert_calls = [s for s in executed_sql if "INSERT INTO" in s]
    assert len(insert_calls) == 2


def test_update_pgvector_empty_pairs_is_noop(jql):
    with patch("jql_embeddings.EMBEDDING_BATCH_SIZE", 32, create=True):
        encode_spy = jql.documentProc._model.encode
        encode_spy.reset_mock()
        jql._update_pgvector_from_annotations([], model_name="all-MiniLM-L6-v2")
    encode_spy.assert_not_called()


# ---------------------------------------------------------------------------
# search_sample_jql_embeddings_db
# ---------------------------------------------------------------------------

def test_search_returns_rows_and_model(jql):
    fake_rows = [(1, "open bugs", "project = TEST AND status = Open", 0.12)]
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = fake_rows
    jql.model.encode.return_value = np.array([0.1, 0.2, 0.3])

    with patch("jql_embeddings.PGVectorClient", return_value=_make_pgclient_mock(mock_cur)):
        rows, returned_model = jql.search_sample_jql_embeddings_db("open bugs", jql.model)

    assert rows == fake_rows
    assert returned_model is jql.model


# ---------------------------------------------------------------------------
# generate_jql — JQL response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_jql_returns_jql(jql):
    fake_rows = [(1, "open bugs", "project = TEST AND status = Open", 0.1)]

    with patch.object(jql, "search_sample_jql_embeddings_db", return_value=(fake_rows, jql.model)), \
         patch("jql_embeddings._build_prompt", return_value="prompt text", create=True), \
         patch("jql_embeddings.OLLAMA_URL", "http://localhost:11434", create=True), \
         patch("jql_embeddings.OLLAMA_MODEL", "llama3", create=True), \
         patch("jql_embeddings.OLLAMA_TEMPERATURE", 0.0, create=True), \
         patch("jql_embeddings._JQL_TAG", "<<JQL>>", create=True), \
         patch("httpx.AsyncClient") as mock_client_cls:

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "<<JQL>>project = TEST AND status = Open"}
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        text, is_general = await jql.generate_jql("open bugs", jql.model)

    assert is_general is False
    assert text == "project = TEST AND status = Open"


# ---------------------------------------------------------------------------
# generate_jql — general (non-JQL) response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_jql_returns_general_answer(jql):
    fake_rows = [(1, "open bugs", "project = TEST", 0.1)]

    with patch.object(jql, "search_sample_jql_embeddings_db", return_value=(fake_rows, jql.model)), \
         patch("jql_embeddings._build_prompt", return_value="prompt text", create=True), \
         patch("jql_embeddings.OLLAMA_URL", "http://localhost:11434", create=True), \
         patch("jql_embeddings.OLLAMA_MODEL", "llama3", create=True), \
         patch("jql_embeddings.OLLAMA_TEMPERATURE", 0.0, create=True), \
         patch("jql_embeddings._JQL_TAG", "<<JQL>>", create=True), \
         patch("httpx.AsyncClient") as mock_client_cls:

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "JQL is a query language for Jira."}
        mock_response.raise_for_status = MagicMock()
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        text, is_general = await jql.generate_jql("what is JQL?", jql.model)

    assert is_general is True
    assert "JQL" in text
