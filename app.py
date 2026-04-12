"""
app.py — aMind command-line entry point.

Supports two modes, selected via a mutually exclusive flag:

  --query [TEXT]  Interactive REPL loop (no TEXT) or single-shot query (with TEXT).
                  In REPL mode, type a query at the prompt and press Enter.
                  Type 'am quit' or 'am exit' to leave. Press Ctrl+C to abort a query.

  --server        Start the FastAPI server (see server.py) on port 8000.
                  Exposes GET /health and POST /query.

Usage
-----
  uv run python app.py --query               # interactive REPL
  uv run python app.py --query "list bugs"   # single-shot
  uv run python app.py --server
  uv run python app.py --help

Both modes initialise AtlasMind on startup, which seeds the pgvector tables
from the annotation and Jira fields files if their content has changed since
the last run (hash-gated — no redundant re-encoding).
"""
import argparse
import asyncio
import json
import time

from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from core.atlasmind import AtlasMind
from core.field_resolver import ResolvedIntentFields
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL, OLLAMA_MODEL, GROQ_MODEL

console = Console()

BORDER_UP = "[bold yellow]╔═════════════════════════════════════════════════════════════════════════╗[/]\n"
BORDER_DWN = "[bold yellow]╚═════════════════════════════════════════════════════════════════════════╝[/]\n"

HELP = """
## Example Queries

- List open bugs assigned to me
- Show unresolved blockers in project MYAPP ordered by priority
- Which issues have been in progress for more than 7 days?
- List all stories in the current sprint that are not yet started
- Show issues created this week grouped by assignee
- Find all critical issues without a fix version

## Commands

| Command      | Description                        |
|--------------|------------------------------------|
| `am help`    | Show this message                  |
| `am history` | Show query history for this session|
| `am quit`    | Exit (also: `exit`, `quit`, `q`)   |
| `Ctrl+C`     | Interrupt query or exit at prompt  |
"""

_EXIT_COMMANDS = {"am quit", "am exit", "am q", "exit", "quit", "q"}
_HELP_COMMANDS = {"am help", "am ?", "?"}


def _print_banner(llm_backend: str = "ollama") -> None:
    f = Figlet(font="slant")
    llm_model = GROQ_MODEL if llm_backend == "groq" else OLLAMA_MODEL
    console.print(BORDER_UP)
    console.print(f.renderText("AtlasMind"))
    console.print(
        "Ask anything about your Jira project in plain English\n\n"
        f"[dim]LLM backend :[/] [cyan]{llm_backend}[/]\n"
        f"[dim]LLM model   :[/] [cyan]{llm_model}[/]\n"
        f"[dim]Embed model :[/] [cyan]{EMBEDDING_MODEL}[/]\n"
    )
    console.print(BORDER_DWN)


def _compute_display_fields(
    atlasmind: AtlasMind,
    jira_result: dict | None,
) -> tuple[list[str], list[str]] | None:
    """Return (standard_names, intent_names) for a JQL result, or None for general answers."""
    if not jira_result or not atlasmind.field_resolver:
        return None
    resolved: ResolvedIntentFields = jira_result.get(
        "resolved_intent_fields", ResolvedIntentFields()
    )
    std_names = atlasmind.field_resolver.display_names_for_ids(atlasmind.standard_field_ids)
    return std_names, resolved.display_names


def _print_result(
    llm_result,
    jira_result: dict | None,
    elapsed: float | None = None,
    display_fields: tuple[list[str], list[str]] | None = None,
) -> None:
    route = "JQL pipeline" if llm_result.jql else "General answer"
    console.print(Rule(style="dim cyan"))
    console.print(f"[dim]Route   : {route}[/]")
    if llm_result.jql:
        console.print(f"\n[bold cyan]JQL[/]    : {llm_result.jql}")
        if llm_result.chart_spec:
            console.print(f"[bold cyan]Chart[/]  : {json.dumps(llm_result.chart_spec)}")
        if jira_result:
            shown = jira_result.get("shown", 0)
            total = jira_result.get("total", 0)
            console.print(f"[bold cyan]Issues[/] : {shown} of {total} returned")
        if display_fields:
            std_names, intent_names = display_fields
            std_str = "  ·  ".join(std_names)
            if intent_names:
                int_str = "  ·  ".join(intent_names)
                console.print(
                    f"[bold cyan]Fields[/] : {std_str}"
                    f"  [dim cyan]+[/]  [italic cyan]{int_str}[/]"
                )
            else:
                console.print(f"[bold cyan]Fields[/] : {std_str}")
        if llm_result.intent_fields:
            console.print(
                f"[dim]LLM proposed intent : {', '.join(llm_result.intent_fields)}[/]"
            )
    answer = llm_result.answer or ""
    if not llm_result.jql and len(answer) > 300:
        answer = answer[:300].rsplit(" ", 1)[0] + " …"
    console.print(f"[bold cyan]Answer[/] : {answer}\n")
    if elapsed is not None:
        console.print(f"[dim]Response time : {elapsed:.2f}s[/]")
    console.print(Rule(style="dim cyan"))


def build_atlasmind(llm_backend: str = "ollama") -> AtlasMind:
    """Initialise AtlasMind and seed both pgvector tables."""
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    atlasmind = AtlasMind(config, llm_backend=llm_backend)
    atlasmind.run()
    return atlasmind


async def repl(atlasmind: AtlasMind, llm_backend: str = "ollama") -> None:
    """Interactive REPL — keeps prompting until the user exits."""
    _print_banner(llm_backend)
    history: list[str] = []

    while True:
        try:
            user_input = console.input(r"[bold cyan]\[atlasmind]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n[dim]Goodbye.[/]")
            break

        if not user_input:
            continue

        if user_input.lower() in _EXIT_COMMANDS:
            console.print("\n[dim]Goodbye.[/]")
            break

        if user_input.lower() in _HELP_COMMANDS:
            console.print(Markdown(HELP))
            continue

        if user_input.lower() == "am history":
            if not history:
                console.print("  [dim]No queries yet.[/]")
            else:
                for i, q in enumerate(history, 1):
                    console.print(f"  [dim]{i}.[/] {q}")
            continue

        history.append(user_input)
        console.print()

        try:
            t0 = time.monotonic()
            llm_result, jira_result = await atlasmind.generate_jql(user_input)
            elapsed = time.monotonic() - t0
            display_fields = _compute_display_fields(atlasmind, jira_result)
            _print_result(llm_result, jira_result, elapsed, display_fields)
        except KeyboardInterrupt:
            console.print("\n[dim][interrupted][/]")
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")

        console.print()


async def run_query(atlasmind: AtlasMind, query: str) -> None:
    """Run a single query and print the JQL and answer."""
    llm_result, jira_result = await atlasmind.generate_jql(query)
    if llm_result.jql:
        print(f"JQL    : {llm_result.jql}")
        fields = _compute_display_fields(atlasmind, jira_result)
        if fields:
            std_names, intent_names = fields
            all_names = std_names + ([f"+{n}" for n in intent_names] if intent_names else [])
            print(f"Fields : {', '.join(all_names)}")
    print(f"Answer : {llm_result.answer}")


def run_server(host: str = "0.0.0.0", port: int = 8000, llm_backend: str = "ollama") -> None:
    """Start the FastAPI server."""
    import os
    import uvicorn
    os.environ["LLM_BACKEND"] = llm_backend
    uvicorn.run("server:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="app",
        description="aMind — natural language to JQL generator using RAG + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app.py --query                        # interactive REPL (local Ollama)\n"
            "  python app.py --query 'list open bugs'       # single-shot query\n"
            "  python app.py --query --model groq           # REPL using Groq cloud\n"
            "  python app.py --server                       # start FastAPI server\n"
            "  python app.py --server --model groq --port 9000\n"
            "\n"
            "Environment variables:\n"
            "  LLM_BACKEND          ollama | groq  (overrides --model when set)\n"
            "  GROQ_API_KEY         Groq API key for local dev\n"
            "  GROQ_API_KEY_OCID    OCI Vault secret OCID (takes priority over GROQ_API_KEY)\n"
            "  GROQ_MODEL           Groq model name          (default: llama-3.3-70b-versatile)\n"
            "  JQL_LOCAL_MODEL      Ollama model name        (default: qwen2.5:3b-instruct-q4_K_M)\n"
            "  JQL_OLLAMA_URL       Ollama base URL          (default: http://localhost:11434)\n"
            "  DATABASE_URL         PostgreSQL + pgvector connection string\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--query",
        nargs="?",
        const="",
        metavar="TEXT",
        help="Interactive REPL (no TEXT) or single-shot query (with TEXT)",
    )
    mode.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server (GET+POST /query, GET /health)",
    )
    parser.add_argument(
        "--model",
        choices=["ollama", "groq"],
        default="ollama",
        metavar="BACKEND",
        help="LLM backend: 'ollama' (local, default) or 'groq' (cloud)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to in server mode (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on in server mode (default: 8000)",
    )
    args = parser.parse_args()

    if args.server:
        run_server(host=args.host, port=args.port, llm_backend=args.model)
    else:
        atlasmind = build_atlasmind(llm_backend=args.model)
        if args.query:
            asyncio.run(run_query(atlasmind, args.query))
        else:
            try:
                asyncio.run(repl(atlasmind, llm_backend=args.model))
            except KeyboardInterrupt:
                console.print("\n\n[dim]Goodbye.[/]")


if __name__ == "__main__":
    main()
