"""
Central configuration for AtlasMind.

All hardcoded defaults live here. Override any value via the corresponding
environment variable without touching this file.
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent

# -- Ollama / LLM -----------------------------------------------------
OLLAMA_URL         = os.getenv("JQL_OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("JQL_LOCAL_MODEL",  "qwen2.5-coder:7b-instruct")
OLLAMA_TEMPERATURE = float(os.getenv("JQL_OLLAMA_TEMPERATURE", "0"))

# -- pgvector / Embeddings ---------------------------------------------
DATABASE_URL         = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jql_vectordb")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE = 32

# -- JQL Embeddings DB schema ---------------------------------------------------------
JQL_TABLE            = "jql_annotations"
JQL_COL_ANNOTATION   = "annotation"
JQL_COL_JQL          = "jql"
JQL_COL_EMBEDDING    = "embedding"
JQL_SEARCH_LIMIT     = 5

# -- Jira field Embeddings DB schema ---------------------------------------------------------
JIRA_FIELD_TABLE             = "jira_field_annotations"
JIRA_FIELD_COL_DESCRIPTION   = "description"
JIRA_FIELD_COL_EMBEDDING     = "embedding"
JIRA_FIELD_SEARCH_LIMIT      = 1

# Field IDs to exclude from embedding regardless of type.
# Add custom fields that are internal, deprecated, or irrelevant to JQL queries.
JIRA_FIELD_IGNORE_IDS: set[str] = {}

# -- JQL annotation file -----------------------------------------------
DEFAULT_ANNOTATION_FILE = os.getenv(
    "JQL_ANNOTATION_FILE",
    str(_ROOT / "data" / "jira-jql-annotated-queries.md"),
)

# -- Jira query defaults -----------------------------------------------
DEFAULT_JQL  = "statusCategory != Done ORDER BY created DESC"
MAX_RESULTS  = 10

# -- Atlassian Rovo MCP ------------------------------------------------
ROVO_MCP_URL = "https://mcp.atlassian.com/v1/mcp"

# -- OAuth 2.1 ---------------------------------------------------------
OAUTH_REDIRECT_URI = "http://localhost:3334/oauth/callback"
OAUTH_SCOPES       = ["search:rovo:mcp", "read:me", "read:account", "offline_access"]
OAUTH_AUTH_URL     = "https://auth.atlassian.com/authorize"
OAUTH_TOKEN_URL    = "https://auth.atlassian.com/oauth/token"
OAUTH_ENV_FILE     = ".env"
