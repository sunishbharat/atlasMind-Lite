# Contributing to AtlasMind

Thanks for taking the time to contribute. Every improvement is welcome - from a single annotation example to a full new feature.

AtlasMind converts natural language into Jira Query Language (JQL) using a RAG pipeline backed by pgvector and multiple LLM backends. It returns structured JSON containing a JQL query, plain-text answer, and chart specification. Contributions are welcome across a wide range - from adding JQL annotation examples (no code required) to implementing new capabilities like conversation memory, shareable query links, multi-source search, and user authentication.

---

## Ways to contribute

Ordered from lowest to highest barrier. Start wherever fits your time.

New to the project? Issues tagged `good first issue` are self-contained, have a clear expected outcome, and do not require deep knowledge of the RAG pipeline or LLM backends. Adding JQL annotation examples, fixing a specific prompt rule, or writing a missing test all fit this category.

If you have questions or get stuck, open an issue with the `question` label.

### 1. Add JQL annotation examples (no coding required)

The file `data/jira_jql_annotated_queries.md` is the most impactful thing you can improve. It is the few-shot example store that the RAG pipeline retrieves from at query time - more diverse, well-written examples directly improve JQL generation quality for everyone.

See [Annotation file format](#annotation-file-format) below for the exact format rules.

Good examples to add:
- Queries using date functions (`startOfWeek()`, `startOfMonth()`, `endOfMonth()`)
- Sprint-scoped queries (`sprint in openSprints()`, `sprint = "Team Name Sprint 3"`)
- Queries combining multiple custom field types
- Queries with `ORDER BY` on non-default fields
- Queries using `IS EMPTY` / `IS NOT EMPTY` for optional fields
- Queries with `CHANGED`, `WAS`, or `WAS IN` for system fields (status, assignee, priority)
- Any query type that is currently underrepresented or that the model gets wrong

### 2. Improve system and router prompts

The system prompt (`config/system_prompt.md`) contains JQL rules and the JSON output schema.
The router prompt (`config/router_prompt.md`) classifies queries as `jql`, `general`, or `raw`.

Useful improvements:
- Add or tighten JQL rules where the model makes systematic mistakes
- Add few-shot examples to the router for query types it misclassifies
- Improve the `assignee`/`reporter` guidance (current gap: LLM uses team names instead of usernames)

### 3. Report bugs

Open a GitHub issue with:
- The natural language query you sent
- The JQL that was generated
- The Jira error or wrong result you received
- Your LLM backend and model name

The more specific the better. A query that reliably fails is more actionable than a general description.

### 4. Implement roadmap features

The roadmap has two tiers. **Tier 1** items have full design documents in `docs/` - sequence diagrams, data models, file-level plans - and are ready for implementation. **Tier 2** items are agreed priorities where the design work itself is the first contribution needed.

#### Tier 1 - Design documents ready, implementation open

| Feature | Design doc | Effort |
|---------|------------|--------|
| JQL Semantic Validator - post-generation field/value correction via embedding similarity | `docs/jql-semantic-validator-design.md` | ~3 days |
| Field Concept Extraction - pre-generation LLM extraction + per-concept vector search | `docs/field-extraction-design.md` | ~4–6 days |
| Changelog-based field change search - async httpx + DuckDB filter | `docs/claude_design_proposal.md` | ~3–4 days |
| Sprint velocity - Jira Agile API + DuckDB aggregation | `docs/claude_design_proposal.md` | ~3 days |

Read the relevant doc before starting. Open a draft PR early to avoid duplicate work.

#### Tier 2 - Design phase: help scope these first

These features are priorities without a design document yet. The first contribution for any of them is a design doc in `docs/` - architecture decisions, affected files, data models, and open questions. Open an issue to discuss before writing.

#### Conversation memory

Every query is currently stateless. Each call to `POST /query` has no knowledge of previous queries in the same session. The feature is: maintain a conversation history per session and inject recent turns into the LLM prompt so the user can ask follow-up questions like "now filter those by assignee" or "show only the last 30 days".

Relevant files: `core/atlasmind.py` (`generate_jql`), `core/models.py` (`QueryRequest`, `QueryResponse`), `server.py`.
Likely additions: `conversation_id` field on requests, a `conversations` PostgreSQL table, a `core/conversation.py` history manager, prompt injection logic in `_build_prompt()`.

#### Shareable query bookmarks

When a query returns results, there is currently no way to share that exact view with another user. The feature is: a `POST /bookmark` endpoint that persists the `QueryResponse` JSON and returns a short unique ID, and a `GET /bookmark/{id}` endpoint that retrieves and replays it. The frontend renders a "copy link" button.

Relevant files: `server.py`, `core/models.py`. Likely addition: `bookmarks` table in PostgreSQL (`id`, `query`, `response_json`, `created_at`, optional `expires_at`).

#### Chart and result export

The chart specification (`ChartSpec`) is returned as JSON and rendered client-side by the frontend using ECharts. There is no way to export the rendered chart or the result table as a file. The feature covers at minimum PNG export and ideally PDF and CSV.

Two valid approaches exist and the design doc should choose one:
- **Frontend-side**: use ECharts' built-in `getDataURL()` for image export and a library such as jsPDF for PDF. Zero new backend dependencies.
- **Backend-side**: a new `POST /export` endpoint that renders the chart server-side (e.g. using `matplotlib` or a headless browser) and returns a file. More portable but heavier.

CSV export of the issues table is backend-only and straightforward: stream the `issues` list from `QueryResponse` as a CSV file.

#### Confluence and general web page search

The current pipeline is Jira-only. The feature is: extend the query router to recognise Confluence queries and route them to a new `ConfluenceSearchClient`, and optionally support fetching and summarising arbitrary web URLs.

Confluence has its own query language (CQL - Confluence Query Language), a REST API (`/rest/api/content/search?cql=...`), and similar concepts to Jira (spaces, pages, labels). A parallel RAG pipeline seeded with Confluence space/page metadata is the natural fit.

Relevant files: `core/router.py` (new route type `cql`), `jira/jira_search.py` (pattern to follow), `rag/` (new embedding store for Confluence content). The `QueryRouter` and `AtlasMind` orchestrator would need new branches analogous to the existing `jql` path.

General web page reading (fetch URL → extract text → summarise) is a separate, simpler capability: a new `web` route type that calls an HTTP client and passes the page content as context to the LLM.

#### User authentication and registration

The server currently accepts Jira credentials per-request via `X-Jira-Token` and `X-Jira-Url` headers - fully stateless. The feature is: add a registration and login flow so users have persistent accounts, their Jira credentials are stored server-side (encrypted), and requests are authenticated via a session token or JWT rather than per-request headers.

This is the most architecturally significant item on this list. Key design questions to resolve in the doc:
- Single-tenant (one org) or multi-tenant (each user has their own Jira credentials)?
- JWT or server-side sessions?
- How to encrypt stored Jira PATs at rest
- Whether to use an existing library (e.g. `fastapi-users`) or implement the auth layer directly

Relevant files: `server.py`, `core/jira_auth.py`, `core/models.py`. Likely additions: `users` table, `auth/` module, registration and login endpoints, JWT middleware.

### 5. Add or improve tests

The test suite lives in `tests/`. Areas with known gaps:

- `tests/test_jql_sanitizer.py` - Pass 7 (Assets `aqlFunction` rewrite) has no tests
- `tests/test_jql_sanitizer.py` - the `where_fields` → `hint_asset_ids` wiring path is untested
- Any new feature you implement should come with tests for the happy path and at least one edge case

Run tests with:

```bash
uv run python -m pytest tests/ -v
```

---

## Development setup

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) - the only supported package manager for this project
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension
- One LLM backend (Ollama is the easiest for local development)

### Install

```bash
uv sync
```

### Minimum local stack (Ollama)

```bash
# Start PostgreSQL with pgvector (example using Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16

# Pull a model into Ollama
ollama pull qwen2.5:3b-instruct-q4_K_M

# Fetch Jira field metadata once (requires a configured profile in config/profiles.json)
uv run python -c "from jira.jira_field_api import fetch_and_save_fields; fetch_and_save_fields()"

# Start the server
uv run python app.py --server
```

### Environment variables

All settings are in `settings.py` and overridable via environment variable. The most important ones for local development:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/jql_vectordb` | pgvector connection |
| `LLM_BACKEND` | `ollama` | Active backend: `ollama`, `groq`, `vllm`, `claude`, `bedrock` |
| `ANTHROPIC_API_KEY` | - | Required for `--model claude` |
| `GROQ_API_KEY` | - | Required for `--model groq` |

See `settings.py` and the README for the full list.

---

## Code conventions

Following these keeps PRs clean and reduces review iterations.

**Package manager:** Always use `uv`. Never use `pip` directly.

```bash
uv add <package>          # add a dependency
uv sync                   # install/sync after pulling
uv run python -m pytest   # run tests
```

**No emojis.** Not in source code, comments, log messages, docstrings, or documentation.

**Generic placeholder names in all examples.** Use names like `Sample Domain`, `Sample Object`, `Sample Project` and keys like `ABCD-1234`, `XY-999`. Do not use names that could be mistaken for real data.

**No new CLI flags in `app.py`.** New optional behaviour belongs behind an environment variable toggle in `settings.py`. The CLI interface (`app.py`) should not grow new `argparse` arguments.

**No GPU system or hardware names in code or docs.** Refer to infrastructure generically (e.g. "GPU inference server") rather than by hostname or model name.

**Comments only when the why is non-obvious.** Do not explain what the code does - use that effort to write clear names instead. A comment should describe a hidden constraint, a subtle invariant, or a workaround for a specific external bug.

---

## Annotation file format

`data/jira_jql_annotated_queries.md` uses a specific block format that the parser depends on.
Violating these rules causes the integrity test to fail.

**Correct format:**

```
/* 501. Show open high priority bugs created this week */
issuetype = Bug AND priority = High AND status != Done AND created >= startOfWeek()
```

**Rules:**

1. The annotation comment `/* N. description */` must be on its own line.
2. The JQL must start on the **next line** - not on the same line as `*/`.
3. No trailing whitespace after `*/`.
4. No un-numbered section header comments (`/* Section Name */`) between numbered blocks.
5. The number `N` must be exactly one more than the previous block. Do not skip numbers.
6. Use generic names in examples - never real project keys, usernames, or org-specific values.

**After editing the file, run the integrity test:**

```bash
uv run python -m pytest tests/test_jql_embeddings.py::test_annotation_file_pair_count_matches_last_annotation_number -v
```

This test counts all `/* N. */` blocks and asserts the total equals the last annotation number. If it fails, the error output shows how many entries were merged and why.

---

## Pull request process

1. Fork the repository and create a branch from `main`.
   - Suggested naming: `feat/<short-description>`, `fix/<short-description>`, `docs/<short-description>`, `test/<short-description>`
2. Keep the PR focused. One logical change per PR.
3. If your change touches the JQL annotation file, run the integrity test before pushing.
4. Write a clear PR description: what the change does, why it is needed, and how you tested it.
5. Reference any related issue number in the PR description.

There is no formal review SLA, but PRs that are small, well-described, and include tests move fastest.

---

## Project structure (quick reference)

```
core/           - orchestrator, router, sanitizer, field resolver, LLM clients
rag/            - pgvector embedding stores (JQL examples, fields, values, assets)
jira/           - Jira REST API clients (search, field fetch, assets, compute)
config/         - system prompt, router prompt, Jira profiles, asset field config
data/           - JQL annotation file, cached Jira field JSON
cloud/          - OCI Vault, TLS/cert handling, config fetcher
tests/          - test suite
docs/           - architecture reference and feature design documents
settings.py     - all defaults; every value overridable via env var
```

The full architecture with data flow, sequence diagrams, and component descriptions is in `docs/architecture.md`.
