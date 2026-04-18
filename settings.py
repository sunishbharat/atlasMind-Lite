"""
Central configuration for AtlasMind.

All hardcoded defaults live here. Override any value via the corresponding
environment variable without touching this file.
"""

import os
from pathlib import Path

_ROOT = Path(__file__).parent

# -- LLM backend selection --------------------------------------------
# "ollama" (default) or "groq"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# -- Ollama / LLM -----------------------------------------------------
OLLAMA_URL         = os.getenv("JQL_OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("JQL_LOCAL_MODEL",  "qwen2.5:3b-instruct-q4_K_M")
OLLAMA_TEMPERATURE    = float(os.getenv("JQL_OLLAMA_TEMPERATURE",  "0.1"))
OLLAMA_TIMEOUT        = int(os.getenv("JQL_OLLAMA_TIMEOUT",        "120"))
OLLAMA_NUM_CTX        = int(os.getenv("JQL_OLLAMA_NUM_CTX",        "2048"))
OLLAMA_NUM_PREDICT    = int(os.getenv("JQL_OLLAMA_NUM_PREDICT",    "512"))
OLLAMA_NUM_THREAD     = int(os.getenv("JQL_OLLAMA_NUM_THREAD",     "4"))
OLLAMA_NUM_BATCH      = int(os.getenv("JQL_OLLAMA_NUM_BATCH",      "256"))
OLLAMA_TOP_P          = float(os.getenv("JQL_OLLAMA_TOP_P",        "0.5"))
OLLAMA_TOP_K          = int(os.getenv("JQL_OLLAMA_TOP_K",          "20"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("JQL_OLLAMA_REPEAT_PENALTY", "1.1"))

# -- Groq cloud LLM ---------------------------------------------------
# GROQ_API_KEY_OCID: set this to your OCI Vault secret OCID on cloud deployments.
# GROQ_API_KEY: used as plaintext fallback for local development.
from cloud.oci_vault import resolve_secret
GROQ_API_KEY    = resolve_secret("GROQ_API_KEY_OCID", "GROQ_API_KEY")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
GROQ_TIMEOUT    = int(os.getenv("GROQ_TIMEOUT", "30"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "500"))

# -- pgvector / Embeddings ---------------------------------------------
DATABASE_URL         = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jql_vectordb")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE = 32

# -- JQL Embeddings DB schema ---------------------------------------------------------
JQL_TABLE            = "jql_annotations"
JQL_COL_ANNOTATION   = "annotation"
JQL_COL_JQL          = "jql"
JQL_COL_EMBEDDING    = "embedding"
JQL_SEARCH_LIMIT     = 3

# -- Jira field Embeddings DB schema ---------------------------------------------------------
JIRA_FIELD_TABLE             = "jira_field_annotations"
JIRA_FIELD_COL_DESCRIPTION   = "description"
JIRA_FIELD_COL_EMBEDDING     = "embedding"
JIRA_FIELD_SEARCH_LIMIT      = 5

# Field IDs to exclude from embedding regardless of type.
# Add custom fields that are internal, deprecated, or irrelevant to JQL queries.
JIRA_FIELD_IGNORE_IDS: set[str] = {}

# -- JQL annotation file -----------------------------------------------
DEFAULT_ANNOTATION_FILE = (
    os.getenv("JQL_ANNOTATION_FILE")
    or str(_ROOT / "data" / "jira_jql_annotated_queries.md")
)

# -- Data directory (domain-scoped subdirs are created inside here) ----
DATA_DIR = _ROOT / "data"

# -- File names — change here to rename files project-wide ------------
JIRA_FIELDS_FILENAME         = "jira_fields.json"
JIRA_ALLOWED_VALUES_FILENAME = "jira_allowed_values.json"
SYSTEM_PROMPT_FILE              = str(_ROOT / "config" / "system_prompt.md")
ROUTER_PROMPT_FILE              = str(_ROOT / "config" / "router_prompt.md")
ROUTER_PROMPT_FILE_OLLAMA       = str(_ROOT / "config" / "router_prompt_ollama.md")
CHART_SPEC_PROMPT_FILE          = str(_ROOT / "config" / "chart_spec_prompt.md")

# -- Jira query defaults -----------------------------------------------
DEFAULT_JQL  = "statusCategory != Done ORDER BY created DESC"
MAX_RESULTS  = 500
MAX_JIRA_RESULTS = int(os.getenv("MAX_JIRA_RESULTS", "2000"))

# -- Intent field resolution -------------------------------------------
# Maximum number of extra fields the LLM may propose per query.
MAX_INTENT_FIELDS = int(os.getenv("MAX_INTENT_FIELDS", "5"))

# Desired standard column IDs — always shown in the frontend.
# Validated against jira_fields.json at startup; missing IDs are logged and dropped.
# Override via env: STANDARD_FIELD_IDS=key,summary,assignee,status
_STANDARD_FIELD_IDS_DEFAULT = "key,summary,assignee,priority,issuetype,created,resolutiondate"
STANDARD_FIELD_IDS: list[str] = [
    f.strip() for f in os.getenv("STANDARD_FIELD_IDS", _STANDARD_FIELD_IDS_DEFAULT).split(",") if f.strip()
]


# -- Atlassian Rovo MCP ------------------------------------------------
ROVO_MCP_URL = "https://mcp.atlassian.com/v1/mcp"

# -- OAuth 2.1 ---------------------------------------------------------
OAUTH_REDIRECT_URI = "http://localhost:3334/oauth/callback"
OAUTH_SCOPES       = ["search:rovo:mcp", "read:me", "read:account", "offline_access"]
OAUTH_AUTH_URL     = "https://auth.atlassian.com/authorize"
OAUTH_TOKEN_URL    = "https://auth.atlassian.com/oauth/token"
OAUTH_ENV_FILE     = ".env"
