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
| `LLM_BACKEND` | `ollama` | LLM backend: `ollama`, `groq`, or `vllm` (overrides `--model` when set) |
| `JQL_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `JQL_LOCAL_MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Ollama model to use |
| `JQL_OLLAMA_TIMEOUT` | `120` | Read timeout in seconds for LLM inference |
| `GROQ_API_KEY` | — | Groq API key (local dev) |
| `GROQ_API_KEY_OCID` | — | OCI Vault secret OCID for `GROQ_API_KEY` (takes priority over `GROQ_API_KEY`) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `JQL_ANNOTATION_FILE` | `data/jira_jql_annotated_queries.md` | Path to JQL annotation file |
| `MAX_JIRA_RESULTS` | `2000` | Maximum number of Jira issues fetched per query (paginated automatically) |
| `MAX_INTENT_FIELDS` | `5` | Maximum extra fields the LLM may propose per query |
| `STANDARD_FIELD_IDS` | `key,summary,assignee,priority,issuetype,created,resolutiondate` | Comma-separated list of Jira field IDs always shown in results — override per project or Docker deployment |
| `VLLM_URL` | — | vLLM server base URL (e.g. `http://100.x.x.x:8002`) |
| `VLLM_TIMEOUT` | `240` | Read timeout in seconds for vLLM inference |
| `VLLM_MAX_TOKENS` | — | Max tokens for vLLM responses |
| `VLLM_API_KEY` | — | API key if the vLLM server requires authentication |

## Running the app

All modes are accessed through `app.py`.

### Interactive REPL

```bash
uv run python app.py --query                  # local Ollama (default)
uv run python app.py --query --model groq     # Groq cloud
uv run python app.py --query --model vllm     # vLLM inference server
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
uv run python app.py --server                           # Ollama backend, port 8000
uv run python app.py --server --model groq --port 9000  # Groq backend, port 9000
uv run python app.py --server --model vllm --port 9000  # vLLM backend, port 9000
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

## Routing overrides

The query router automatically classifies each query as JQL or general. If the router misclassifies a query, you can force the route by appending a flag:

| Flag | Effect |
|------|--------|
| `/jql` | Forces the JQL pipeline regardless of LLM classification |
| `/general` | Forces the general answer path, skipping the JQL pipeline |

The flag is stripped from the query before it is sent to the LLM, so it does not affect the generated JQL or answer.

**Examples:**

```
[atlasmind]> how many states are there in India /general

  Route   : General answer
  Answer  : What is the definition of an atom?

[atlasmind]> list issues in KAFKA /jql

  Route   : JQL pipeline
  JQL     : project = KAFKA ORDER BY created DESC
```

Overrides work across all LLM backends (Ollama, Groq, and vLLM).

## Architecture

**Data flow:**

1. `JQL_Embeddings.run()` seeds pgvector with `(annotation, JQL)` pairs parsed from the annotation file
2. `Jira_Field_Embeddings.run()` seeds pgvector with Jira field metadata (name, type, allowed values) — auto-fetched from the Jira REST API on first run if the file is absent
3. At query time, `QueryRouter` makes a single fast LLM call to classify the query:
   - **General query** → answered immediately; no embeddings or Jira API calls
   - **JQL query** → full RAG pipeline: encode → similarity search → prompt → LLM → Jira API
4. The assembled prompt (system instructions + fields + examples + query) is sent to the active LLM (Ollama, Groq, or vLLM)
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
| `core/vllm_client.py` | Async vLLM REST client (OpenAI-compatible); auto-detects model from `/v1/models`; used when `--model=vllm` |
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

## Running vLLM on a GPU system (GPU inference server)

AtlasMind on OCI A1 can offload all LLM inference to a local GPU system over Tailscale. Only vLLM needs to run on the GPU system — no database, no AtlasMind installation required there.

### What runs where

| Machine | What runs |
|---------|-----------|
| GPU system | vLLM only — serves the model over HTTP |
| OCI A1 (always-on) | AtlasMind + Postgres + Ollama (fallback) + frontend |

AtlasMind on OCI A1 sends prompts to vLLM on the GPU system over Tailscale. When the GPU system is off, AtlasMind falls back to its local Ollama automatically.

### Step 1 — Install WSL2 (Windows only)

vLLM does not run natively on Windows. You need WSL2 with Ubuntu.

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart when prompted. After restart, Ubuntu opens and asks you to create a username and password. This is your Linux environment — all remaining steps run inside WSL2.

To open WSL2 later: search for **Ubuntu** in the Start menu, or run `wsl` in any terminal.

### Step 2 — Verify the GPU is visible in WSL2

The NVIDIA driver is automatically bridged from Windows into WSL2 — no separate CUDA toolkit installation needed. vLLM's pip package bundles the CUDA runtime libraries it needs.

Run inside WSL2:

```bash
nvidia-smi
```

You should see your GPU listed with driver version and VRAM. If this command fails, reinstall the latest NVIDIA driver on Windows first, then retry.

### Step 3 — Install vLLM in a virtual environment

Ubuntu 24.04 does not allow system-wide pip installs. Use a virtual environment:

```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

After activation you will see `(vllm-env)` in your prompt. This download is large (~5 GB) — let it complete fully before continuing.

Always activate the environment before running vLLM in future sessions:

```bash
source ~/vllm-env/bin/activate
```

### Step 4 — Choose and run a model

With 8 GB VRAM, use a quantized 7B model. AWQ quantization gives the best quality-to-size ratio and is natively supported by vLLM.

Recommended for AtlasMind (reliable structured JSON and JQL output):

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port 8002 \
  --host 0.0.0.0
```

`Qwen2.5-Coder` is preferred over the general instruct variant because JQL is a query language (similar to SQL). The Coder model is trained on code and structured DSLs, making it more reliable at generating syntactically correct JQL and strictly following the JSON output format (`jql`, `intent_fields`, `chart_spec`, `answer`).

`--gpu-memory-utilization 0.85` reserves 85% of VRAM for vLLM. The default is 0.9 (90%) which can exceed available VRAM on 8 GB cards due to Windows/WSL2 overhead. Lower to `0.80` if startup still fails.

`--max-model-len 8192` caps the context window at 8192 tokens. The model's default (32768) requires more KV cache than fits in 8 GB after loading weights. 8192 is sufficient for AtlasMind — typical prompts (system prompt + RAG examples + query) are 1500–2500 tokens.

On first run, this downloads the model weights from HuggingFace (~4.5 GB). Subsequent runs load from the local cache. Wait until you see:

```
INFO:     Application startup complete.
```

The server is now listening on port 8002.

**Alternative models** (all fit in 8 GB VRAM with AWQ):

| Model | VRAM | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | ~4.5 GB | General instruct, solid fallback |
| `meta-llama/Llama-3.1-8B-Instruct-AWQ` | ~5.5 GB | Strong reasoning, good alternative |

### Step 5 — Verify the server is running

From WSL2, confirm the API responds:

```bash
curl http://localhost:8002/v1/models
```

You should see a JSON response listing the loaded model name.

### Step 6 — Configure AtlasMind on OCI A1

On the OCI A1 machine, set the following environment variables before starting AtlasMind:

```bash
export VLLM_URL=http://<gpu-system-tailscale-ip>:8002
```

Replace `<gpu-system-tailscale-ip>` with the GPU system's Tailscale IP address (find it by running `tailscale ip` in PowerShell on the GPU system, or clicking the Tailscale tray icon).

Then start AtlasMind with the vLLM backend:

```bash
uv run python app.py --server --model vllm
```

AtlasMind auto-detects the loaded model from vLLM's `/v1/models` endpoint — no need to set the model name explicitly.

### Keeping vLLM running across WSL2 sessions

WSL2 shuts down when you close the terminal. To keep vLLM running in the background:

```bash
source ~/vllm-env/bin/activate
nohup vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port 8002 \
  --host 0.0.0.0 > ~/vllm.log 2>&1 &
```

Logs go to `~/vllm.log`. Check them with `tail -f ~/vllm.log`.

---

## Setting up Tailscale for vLLM access

Tailscale creates a private network between your GPU system and OCI A1, so AtlasMind can reach vLLM securely without exposing any ports to the internet.

### Step 1 — Install Tailscale on Windows

Download and install Tailscale from [tailscale.com/download](https://tailscale.com/download). Run the installer and sign in with your Tailscale account (Google, GitHub, or Microsoft login).

Once signed in, Tailscale assigns your Windows machine a private IP in the `100.x.x.x` range. You will see the Tailscale icon in the system tray.

### Step 2 — Configure WSL2 networking

Edit (or create) `C:\Users\<username>\.wslconfig` and add:

```ini
[wsl2]
networkingMode=mirrored
firewall=false
```

Restart WSL2 to apply:

```powershell
wsl --shutdown
```

`networkingMode=mirrored` makes WSL2 share the Windows network stack directly — vLLM is reachable at the Windows machine's IP without any port proxy. `firewall=false` disables the WSL2 Hyper-V firewall layer, which otherwise blocks inbound connections independently of other firewall rules.

### Step 3 — Configure the Windows firewall

Add an inbound allow rule for TCP port 8002.

**If you use Windows Defender Firewall only:**

1. Press `Win + R` → type `wf.msc` → Enter
2. Inbound Rules → New Rule → Port → TCP → 8002 → Allow the connection → All profiles → Finish

**If you use a third-party firewall suite (e.g. Norton 360, McAfee):**

Third-party firewall suites include their own firewall engine that runs alongside Windows Defender Firewall. Add the port 8002 allow rule in your firewall suite's settings — for Norton: Settings → Firewall → Traffic Rules → Add → Action: Allow, Direction: Inbound, Protocol: TCP, Local port: 8002, Profile: All.

> **Note:** If inbound connections are still blocked after adding the rule, both firewall engines may be active simultaneously and conflicting. If your third-party suite is the intended firewall, disable Windows Defender Firewall so only one engine is enforcing rules. Run the following in PowerShell as Administrator:
>
> ```powershell
> Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
> ```
>
> This disables Windows Defender Firewall across all profiles. Your third-party firewall (Norton, McAfee, etc.) remains active.

### Step 4 — Restarting vLLM after WSL2 shutdown

WSL2 resets completely on every shutdown (`wsl --shutdown`, PC restart, or closing the terminal) — all running processes including vLLM are killed. You must restart vLLM each time WSL2 comes back up.

To make this less tedious, add a shell alias to your `~/.bashrc`:

```bash
echo "alias start-vllm='source ~/vllm-env/bin/activate && vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --quantization awq --gpu-memory-utilization 0.85 --max-model-len 8192 --port 8002 --host 0.0.0.0'" >> ~/.bashrc
source ~/.bashrc
```

Then to start vLLM in any future session:

```bash
start-vllm
```

Or in the background:

```bash
start-vllm > ~/vllm.log 2>&1 &
```

Then follow the logs:

```bash
tail -f ~/vllm.log
```

### Step 5 — Find your Tailscale IP (on the GPU system)

In PowerShell on Windows, run:

```powershell
tailscale ip
```

Or click the Tailscale system tray icon — your IP is shown at the top. It will look like `100.x.x.x`.

Note this IP — you will set it as `VLLM_URL` on OCI A1.

### Step 6 — Install Tailscale on OCI A1

On the OCI A1 instance, run:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Follow the authentication link printed in the terminal to connect OCI A1 to the same Tailscale account. Once authenticated, OCI A1 and your GPU system are on the same private network.

### Step 7 — Verify connectivity

From OCI A1, confirm it can reach vLLM on the GPU system (replace with your actual Tailscale IP):

```bash
curl http://100.x.x.x:8002/v1/models
```

You should get back a JSON response listing the loaded model. If the request times out, check that:
- vLLM is running in WSL2 with `--host 0.0.0.0`
- `.wslconfig` has `networkingMode=mirrored` and `firewall=false`, and WSL2 was restarted after the change
- Both machines show as **Connected** in the Tailscale admin console at [login.tailscale.com](https://login.tailscale.com)
- The firewall allow rule for port 8002 is in place (Step 3)
- If using a third-party firewall suite, check whether both firewall engines are conflicting (see Step 3 note)

### Step 8 — Configure AtlasMind

On OCI A1, set the Tailscale IP before starting the server:

```bash
export VLLM_URL=http://100.x.x.x:8002
uv run python app.py --server --model vllm
```

---

## Running tests

```bash
uv run python -m pytest tests/ -v
```
