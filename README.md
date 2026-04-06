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

## Running the app

All modes are accessed through `app.py`.

### Interactive REPL

```bash
uv run python app.py --query
```

Starts a Rich terminal loop with the AtlasMind banner. Type a natural language query and press Enter to get JQL and an answer.

```
[atlasmind]> list open bugs assigned to me

  JQL    : assignee = currentUser() AND issuetype = Bug AND status != Done ORDER BY created DESC
  Chart  : {"type": "bar", "x_field": "status", "y_field": "count", "title": "Open bugs by status"}
  Answer : Open bugs currently assigned to you
```

**REPL commands:**

| Command | Description |
|---------|-------------|
| `am help` | Show example queries and command list |
| `am history` | Show query history for this session |
| `exit` / `quit` / `q` / `am quit` | Exit the REPL |
| `Ctrl+C` at prompt | Exit cleanly |
| `Ctrl+C` during query | Interrupt the current query, return to prompt |

### Single-shot query

```bash
uv run python app.py --query "list open bugs assigned to me"
```

Runs one query, prints `JQL` and `Answer`, then exits. Useful for scripting.

### FastAPI server

```bash
uv run python app.py --server
```

Starts the REST API on `http://0.0.0.0:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/query` | Generate JQL from natural language |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list open bugs assigned to me"}'
```

```json
{
  "jql": "assignee = currentUser() AND issuetype = Bug AND status != Done ORDER BY created DESC",
  "chart_spec": {"type": "bar", "x_field": "status", "y_field": "count", "title": "Open bugs by status"},
  "answer": "Open bugs currently assigned to you"
}
```

## Architecture

**Data flow:**

1. `JQL_Embeddings.run()` seeds pgvector with `(annotation, JQL)` pairs parsed from the annotation file
2. `Jira_Field_Embeddings.run()` seeds pgvector with Jira field metadata (name, type, allowed values) — auto-fetched from the Jira REST API on first run if the file is absent
3. At query time, `AtlasMind._build_prompt()` retrieves the top-N most semantically similar JQL examples and Jira fields via vector similarity search
4. The assembled prompt (system instructions + fields + examples + query) is sent to Ollama
5. Ollama returns structured JSON with `jql`, `chart_spec`, and `answer`

Both seeding steps are hash-gated — re-encoding is skipped if the source files have not changed since the last run.

**Jira fields are stored per domain** under `data/{domain_slug}/` (e.g. `data/issues_apache_org/jira_fields.json`). Switching the active profile in `config/profiles.json` automatically uses the correct set of files for that Jira instance.

**Key files:**

| File | Role |
|------|------|
| `app.py` | CLI entry point — `--query` (REPL / single-shot) and `--server` modes |
| `server.py` | FastAPI app with `/health` and `/query` endpoints |
| `atlasmind.py` | Top-level orchestrator — `run()` seeds both DBs, `generate_jql()` is the query entry point |
| `jql_embeddings.py` | Seeds and searches the JQL annotation pgvector table |
| `jira_field_embeddings.py` | Seeds and searches the Jira field metadata pgvector table |
| `jira_field_api.py` | Fetches field metadata and allowed values from the Jira REST API |
| `ollama_client.py` | Sync `test_connection()` and async `generate_jql()` against the Ollama API |
| `seed_manager.py` | MD5 hash-based seeding gate stored in a `seed_metadata` pgvector table |
| `config/profiles.json` | Jira connection profiles (URL, credentials); `default` key selects the active one |
| `settings.py` | All defaults and file names, overridable via environment variables |

## Jira connection profiles

Edit `config/profiles.json` to configure your Jira instance:

```json
{
  "default": "work",
  "profiles": {
    "work": {
      "jira_url": "https://issues.apache.org/jira",
      "email": "",
      "token": "",
      "jira_type": "server"
    },
    "personal": {
      "jira_url": "https://myorg.atlassian.net",
      "email": "me@example.com",
      "token": "my-api-token"
    }
  }
}
```

Change `"default"` to switch the active instance. Jira fields are auto-fetched and stored in `data/{domain_slug}/` on first run.

## Response model

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

### Jira fields (`data/{domain_slug}/jira_fields.json`)

Fetched automatically on first run from `/rest/api/2/field`. Keyed by field ID. A companion `jira_allowed_values.json` is also fetched and merged in to enrich descriptions with discrete option lists (e.g. status values, issue types).

## Running tests

```bash
uv run python -m pytest tests/ -v
```
