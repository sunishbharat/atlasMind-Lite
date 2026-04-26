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
OLLAMA_TIMEOUT        = int(os.getenv("JQL_OLLAMA_TIMEOUT",        "240"))
OLLAMA_NUM_CTX        = int(os.getenv("JQL_OLLAMA_NUM_CTX",        "2048"))
OLLAMA_NUM_PREDICT    = int(os.getenv("JQL_OLLAMA_NUM_PREDICT",    "512"))
OLLAMA_NUM_THREAD     = int(os.getenv("JQL_OLLAMA_NUM_THREAD",     "4"))
OLLAMA_NUM_BATCH      = int(os.getenv("JQL_OLLAMA_NUM_BATCH",      "256"))
OLLAMA_TOP_P          = float(os.getenv("JQL_OLLAMA_TOP_P",        "0.5"))
OLLAMA_TOP_K          = int(os.getenv("JQL_OLLAMA_TOP_K",          "20"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("JQL_OLLAMA_REPEAT_PENALTY", "1.1"))

# -- vLLM local inference server ---------------------------------------
# VLLM_MODEL: auto-detected from /v1/models if left empty.
# VLLM_API_KEY: only needed if vLLM was started with --api-key.
VLLM_URL         = os.getenv("VLLM_URL",          "http://localhost:8002")
VLLM_MODEL       = os.getenv("VLLM_MODEL",         "")
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
VLLM_TIMEOUT     = int(os.getenv("VLLM_TIMEOUT",    "240"))
VLLM_MAX_TOKENS  = int(os.getenv("VLLM_MAX_TOKENS", "500"))
VLLM_API_KEY     = os.getenv("VLLM_API_KEY",        "")
VLLM_FALLBACK    = os.getenv("VLLM_FALLBACK",       "ollama")

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

JQL_RETRY_TEMPLATE = (
    "\n\nRETRY: your previous JQL was rejected by Jira.\n"
    "  Bad JQL : {bad_jql}\n"
    "  Error   : {error}\n\n"
    "Generate corrected JQL. Return the same JSON format. Do not repeat the same mistake."
)

# Used when the Jira error identifies a specific invalid field by name.
# More directive than JQL_RETRY_TEMPLATE — suited for small models that need
# explicit instruction rather than general guidance.
JQL_RETRY_FIELD_TEMPLATE = (
    "\n\nRETRY: '{field}' is not a valid JQL field.\n"
    "  Bad JQL : {bad_jql}\n\n"
    "Remove the entire condition containing '{field}' from the JQL.\n"
    "Do not quote it, do not rename it — remove it entirely.\n"
    "Return corrected JQL in the same JSON format."
)

# Used when multiple invalid fields are identified in one Jira error response.
JQL_RETRY_FIELDS_TEMPLATE = (
    "\n\nRETRY: multiple invalid fields in the JQL.\n"
    "  Bad JQL : {bad_jql}\n"
    "  Invalid fields: {fields}\n\n"
    "Remove ALL conditions containing these fields entirely.\n"
    "Do not rename or quote them — remove each condition entirely.\n"
    "Return corrected JQL in the same JSON format."
)

# -- Jira query defaults -----------------------------------------------
DEFAULT_JQL  = "statusCategory != Done ORDER BY created DESC"
MAX_RESULTS  = 1000
MAX_JIRA_RESULTS = int(os.getenv("MAX_JIRA_RESULTS", "2000"))

# -- JQL retry -----------------------------------------------------------
# Total attempts per query: 1 initial + (JQL_MAX_ATTEMPTS - 1) retries.
JQL_MAX_ATTEMPTS = int(os.getenv("JQL_MAX_ATTEMPTS", "4"))

# -- Intent field resolution -------------------------------------------
# Maximum number of extra fields the LLM may propose per query.
MAX_INTENT_FIELDS = int(os.getenv("MAX_INTENT_FIELDS", "5"))

# Desired standard column IDs — always shown in the frontend.
# Validated against jira_fields.json at startup; missing IDs are logged and dropped.
# Override via env: STANDARD_FIELD_IDS=key,summary,assignee,status
_STANDARD_FIELD_IDS_DEFAULT = "key,summary,assignee,status,priority,issuetype,created,resolutiondate,project,fixVersion"
STANDARD_FIELD_IDS: list[str] = [
    f.strip() for f in (os.getenv("STANDARD_FIELD_IDS") or _STANDARD_FIELD_IDS_DEFAULT).split(",") if f.strip()
]


# -- Atlassian Rovo MCP ------------------------------------------------
ROVO_MCP_URL = "https://mcp.atlassian.com/v1/mcp"

# -- OAuth 2.1 ---------------------------------------------------------
OAUTH_REDIRECT_URI = "http://localhost:3334/oauth/callback"
OAUTH_SCOPES       = ["search:rovo:mcp", "read:me", "read:account", "offline_access"]
OAUTH_AUTH_URL     = "https://auth.atlassian.com/authorize"
OAUTH_TOKEN_URL    = "https://auth.atlassian.com/oauth/token"
OAUTH_ENV_FILE     = ".env"
