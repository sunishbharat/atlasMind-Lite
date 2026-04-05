# aMind-partial

A JQL (Jira Query Language) generator that translates natural language queries into JQL using RAG (Retrieval-Augmented Generation) with pgvector and a local Ollama LLM.

## Prerequisites

- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension
- [Ollama](https://ollama.ai) running locally with a model loaded (default: `qwen2.5-coder:7b-instruct`)
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
| `JQL_LOCAL_MODEL` | `qwen2.5-coder:7b-instruct` | Ollama model to use |
| `JQL_ANNOTATION_FILE` | `data/jira-jql-annotated-queries.md` | Path to JQL annotation file |

## Using `JQL_Embeddings`

### 1. Seed the database

Call `run()` once on startup. It creates the pgvector schema and loads the annotation file into the database. Re-seeding is skipped automatically if the annotation file hasn't changed (hash-checked).

```python
from pathlib import Path
from jql_embeddings import JQL_Embeddings
from dconfig import EmbeddingsConfig

config = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")
jql = JQL_Embeddings(config)
model = jql.run(Path("data/jira-jql-annotated-queries.md"))
```

`run()` returns the loaded `SentenceTransformer` model for reuse in search/generation.

### 2. Search for similar JQL examples

Encode a natural language query and retrieve the top-5 nearest annotation/JQL pairs from pgvector:

```python
rows, model = jql.search_sample_jql_embeddings_db("open bugs assigned to me", model)
for id_, annotation, jql_text, distance in rows:
    print(f"{annotation!r} → {jql_text!r}  (dist={distance:.4f})")
```

### 3. Generate JQL from natural language

Send the query through the full RAG pipeline (similarity search → few-shot prompt → Ollama):

```python
import asyncio

text, is_general = asyncio.run(jql.generate_jql("open bugs assigned to me", model))

if is_general:
    print("General answer:", text)
else:
    print("JQL:", text)
```

`is_general=False` means the response is a JQL string ready to send to the Jira REST API.  
`is_general=True` means Ollama returned a plain-text answer (e.g. for questions like "what is JQL?").

### Full example

```python
import asyncio
from pathlib import Path
from jql_embeddings import JQL_Embeddings
from dconfig import EmbeddingsConfig

config = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")
jql = JQL_Embeddings(config)

# Seed DB (skipped automatically if annotation file unchanged)
model = jql.run(Path("data/jira-jql-annotated-queries.md"))

# Generate JQL
text, is_general = asyncio.run(jql.generate_jql("bugs created this week", model))
print(text)
```

## Annotation file format

The annotation file is a Markdown file with `/* comment */\nJQL` pairs:

```
/* open bugs assigned to me */
assignee = currentUser() AND status = Open

/* high priority tickets created this week */
priority = High AND created >= -7d ORDER BY created DESC
```

## Running tests

```bash
uv run python -m pytest tests/ -v
```
