# Atlasmind-Lite

A natural language to JQL (Jira Query Language) generator using RAG (Retrieval-Augmented Generation) with pgvector and a local Ollama LLM or Groq cloud LLM. Returns structured JSON with a JQL query, a chart specification, and a plain-text answer. A two-stage router answers general questions immediately without touching the JQL pipeline.

## Prerequisites

- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.ai) running locally with a model loaded (default: `qwen2.5:3b-instruct-q4_K_M`) — **or** a Groq API key for cloud mode
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
| `LLM_BACKEND` | `ollama` | LLM backend: `ollama` or `groq` (overrides `--model` when set) |
| `JQL_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `JQL_LOCAL_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Ollama model to use |
| `JQL_OLLAMA_TIMEOUT` | `120` | Read timeout in seconds for LLM inference |
| `GROQ_API_KEY` | — | Groq API key (local dev) |
| `GROQ_API_KEY_OCID` | — | OCI Vault secret OCID for `GROQ_API_KEY` (takes priority over `GROQ_API_KEY`) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `JQL_ANNOTATION_FILE` | `data/jira_jql_annotated_queries.md` | Path to JQL annotation file |

## Running the app

All modes are accessed through `app.py`.

### Interactive REPL

```bash
uv run python app.py --query                  # local Ollama (default)
uv run python app.py --query --model groq     # Groq cloud
```

Starts a Rich terminal loop with the AtlasMind banner. Type a natural language query and press Enter to get JQL and an answer.

```
[atlasmind]> list open bugs assigned to me

  Route   : JQL pipeline
  JQL     : assignee = currentUser() AND issuetype = Bug AND status != Done ORDER BY created DESC
  Chart   : {"type": "bar", "x_field": "status", "y_field": "count", "title": "Open bugs by status"}
  Answer  : Open bugs currently assigned to you
  Response time : 2.34s
```

General questions are answered directly without going through the JQL pipeline:

```
[atlasmind]> what is the difference between a bug and a task?

  Route   : General answer
  Answer  : A bug represents a defect or unexpected behaviour in the software...
  Response time : 0.81s
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
uv run python app.py --server                          # Ollama backend, port 8000
uv run python app.py --server --model groq --port 9000 # Groq backend, port 9000
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
3. At query time, `QueryRouter` makes a single fast LLM call to classify the query:
   - **General query** → answered immediately; no embeddings or Jira API calls
   - **JQL query** → full RAG pipeline: encode → similarity search → prompt → LLM → Jira API
4. The assembled prompt (system instructions + fields + examples + query) is sent to the active LLM (Ollama or Groq)
5. LLM returns structured JSON with `jql`, `chart_spec`, and `answer`
6. JQL is post-processed (strip LIMIT, arithmetic ORDER BY), then executed against the Jira REST API

Both seeding steps are hash-gated — re-encoding is skipped if the source files have not changed since the last run.

**Jira fields are stored per domain** under `data/{domain_slug}/` (e.g. `data/issues_apache_org/jira_fields.json`). Switching the active profile in `config/profiles.json` automatically uses the correct set of files for that Jira instance.

**Key files:**

| File | Role |
|------|------|
| `app.py` | CLI entry point — `--query` (REPL / single-shot), `--server`, `--model`, `--host`, `--port` |
| `server.py` | FastAPI app with `/health` and `/query` endpoints |
| `core/atlasmind.py` | Top-level orchestrator — `run()` seeds both DBs, `generate_jql()` is the query entry point |
| `core/router.py` | Two-stage query router — fast LLM classify before triggering RAG pipeline |
| `core/ollama_client.py` | Sync `test_connection()` and async `generate_jql()` against the Ollama API |
| `core/groq_client.py` | Async Groq REST client (OpenAI-compatible); used when `--model=groq` |
| `cloud/oci_vault.py` | OCI Vault secret fetching via Instance Principal; fallback to plain env var |
| `rag/jql_embeddings.py` | Seeds and searches the JQL annotation pgvector table |
| `rag/jira_field_embeddings.py` | Seeds and searches the Jira field metadata pgvector table |
| `jira/jira_field_api.py` | Fetches field metadata and allowed values from the Jira REST API |
| `seed_manager.py` | MD5 hash-based seeding gate stored in a `seed_metadata` pgvector table |
| `config/profiles.json` | Jira connection profiles (URL, credentials); `default` key selects the active one |
| `config/system_prompt.md` | JQL-only system prompt (general answers handled by router) |
| `config/router_prompt.md` | Router prompt template with Jira vocabulary list and few-shot examples |
| `settings.py` | All defaults and env-overridable settings for both Ollama and Groq backends |

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
