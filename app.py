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

from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from atlasmind import AtlasMind, JqlResponse
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL, OLLAMA_MODEL

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


def _print_banner() -> None:
    f = Figlet(font="slant")
    console.print(BORDER_UP)
    console.print(f.renderText("AtlasMind"))
    console.print(
        "Ask anything about your Jira project in plain English\n\n"
        f"[dim]LLM model   :[/] [cyan]{OLLAMA_MODEL}[/]\n"
        f"[dim]Embed model :[/] [cyan]{EMBEDDING_MODEL}[/]\n"
    )
    console.print(BORDER_DWN)


def _print_result(result: JqlResponse) -> None:
    console.print(Rule(style="dim cyan"))
    if result.jql:
        console.print(f"\n[bold cyan]JQL[/]    : {result.jql}")
        if result.chart_spec:
            console.print(f"[bold cyan]Chart[/]  : {json.dumps(result.chart_spec)}")
    console.print(f"[bold cyan]Answer[/] : {result.answer}\n")
    console.print(Rule(style="dim cyan"))


def build_atlasmind() -> AtlasMind:
    """Initialise AtlasMind and seed both pgvector tables."""
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    atlasmind = AtlasMind(config)
    atlasmind.run()
    return atlasmind


async def repl(atlasmind: AtlasMind) -> None:
    """Interactive REPL — keeps prompting until the user exits."""
    _print_banner()
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
            result = await atlasmind.generate_jql(user_input)
            _print_result(result)
        except KeyboardInterrupt:
            console.print("\n[dim][interrupted][/]")
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")

        console.print()


async def run_query(atlasmind: AtlasMind, query: str) -> None:
    """Run a single query and print the JQL and answer."""
    result = await atlasmind.generate_jql(query)
    if result.jql:
        print(f"JQL    : {result.jql}")
    print(f"Answer : {result.answer}")


def run_server() -> None:
    """Start the FastAPI server on http://0.0.0.0:8000."""
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="app",
        description="aMind — natural language to JQL generator",
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
        help="Start the FastAPI server on http://0.0.0.0:8000",
    )
    args = parser.parse_args()

    if args.server:
        run_server()
    else:
        atlasmind = build_atlasmind()
        if args.query:
            asyncio.run(run_query(atlasmind, args.query))
        else:
            try:
                asyncio.run(repl(atlasmind))
            except KeyboardInterrupt:
                console.print("\n\n[dim]Goodbye.[/]")


if __name__ == "__main__":
    main()
