# aMind-partial

A natural language to JQL (Jira Query Language) generator using RAG (Retrieval-Augmented Generation) with pgvector and a local Ollama LLM. Returns structured JSON with a JQL query, a chart specification, and a plain-text answer.

## Prerequisites

- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.ai) running locally with a model loaded (default: `qwen3.5:9b`)
- Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Set the following environment variables (or rely on the defaults in `settings.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/jql_vectordb` | pgvector connection string |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model name |
| `JQL_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `JQL_LOCAL_MODEL` | `qwen3.5:9b` | Ollama model to use |
| `JQL_OLLAMA_TIMEOUT` | `120` | Read timeout in seconds for LLM inference (increase for large prompts) |
| `JQL_ANNOTATION_FILE` | `data/jira_jql_annotated_queries.md` | Path to JQL annotation file |
| `JIRA_FIELDS_FILE` | `data/jira_fields.json` | Path to Jira fields JSON from `/rest/api/2/field` |

## Architecture

**Data flow:**

1. `JQL_Embeddings.run()` seeds pgvector with `(annotation, JQL)` pairs parsed from the annotation file
2. `Jira_Field_Embeddings.run()` seeds pgvector with Jira field metadata (name, type, allowed values) from the fields JSON
3. At query time, `AtlasMind._build_prompt()` retrieves the top-N most semantically similar JQL examples and Jira fields via vector similarity search
4. The assembled prompt (system instructions + fields + examples + query) is sent to Ollama
5. Ollama returns structured JSON with `jql`, `chart_spec`, and `answer`

Both seeding steps are hash-gated — re-encoding is skipped if the source files have not changed since the last run.

**Key files:**

| File | Role |
|------|------|
| `atlasmind.py` | Top-level orchestrator — `run()` seeds both DBs, `generate_jql()` is the query entry point |
| `jql_embeddings.py` | Seeds and searches the JQL annotation pgvector table |
| `jira_field_embeddings.py` | Seeds and searches the Jira field metadata pgvector table |
| `ollama_client.py` | Sync `test_connection()` and async `generate_jql()` against the Ollama API |
| `seed_manager.py` | MD5 hash-based seeding gate stored in a `seed_metadata` pgvector table |
| `dconfig.py` | Pydantic `EmbeddingsConfig` model |
| `settings.py` | All defaults, overridable via environment variables |

## Usage

```python
import asyncio
from atlasmind import AtlasMind
from dconfig import EmbeddingsConfig

config = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")
atlasmind = AtlasMind(config)

# Seed both pgvector tables (skipped if source files unchanged)
atlasmind.run()

# Generate JQL from a natural language query
result = asyncio.run(atlasmind.generate_jql(
    "List open bugs assigned to me, grouped by priority"
))

print(result.jql)         # assignee = currentUser() AND ...
print(result.chart_spec)  # {"type": "bar", "x_field": "priority", ...}
print(result.answer)      # "Open bugs assigned to the current user, grouped by priority"
```

### Response model required for frontend UI

`generate_jql()` returns a `JqlResponse` Pydantic model:

```python
class JqlResponse(BaseModel):
    jql: str | None        # None when the query is not Jira-related
    chart_spec: dict | None
    answer: str
```

For general (non-Jira) questions, `jql` and `chart_spec` are `None` and `answer` contains the plain-text response.

## Data files

### JQL annotation file (`data/jira_jql_annotated_queries.md`)

Markdown file with `/* comment */\nJQL` pairs used as few-shot examples:

```
/* open bugs assigned to me */
assignee = currentUser() AND issuetype = Bug AND status != Done ORDER BY created DESC

/* high priority tickets created this week */
priority = High AND created >= startOfWeek() ORDER BY created DESC
```

### Jira fields file (`data/jira_fields.json`)

JSON from the Jira REST API endpoint `/rest/api/2/field`, keyed by field ID. Used to ground the LLM in the exact field IDs and allowed values available in your Jira instance.

An optional `data/jira_allowed_values.json` (produced by `jira_field_api.py`) is merged in to enrich field descriptions with discrete option lists (e.g. status values, issue types).

## Running tests

```bash
uv run python -m pytest tests/ -v
```
