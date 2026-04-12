"""
core/router.py — fast query router.

QueryRouter makes a single lightweight LLM call to decide whether a query
needs the full RAG + JQL pipeline or can be answered directly.

The router prompt lives in config/router_prompt.md — edit that file to tune
routing behaviour without touching Python code.

User overrides
--------------
Users can force a route by appending a flag to their query:

    /jql     — force JQL pipeline regardless of LLM classification
    /general — force general answer path (LLM still generates the answer)

The flag is stripped from the query before it is sent to the LLM.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_JQL_SIGNAL_RE     = re.compile(r"^jql\b", re.IGNORECASE)
_GENERAL_SIGNAL_RE = re.compile(r"^general\b", re.IGNORECASE)
# Explicit user override flags — matched anywhere in the query
_OVERRIDE_JQL_RE     = re.compile(r"\s*/jql\b", re.IGNORECASE)
_OVERRIDE_GENERAL_RE = re.compile(r"\s*/general\b", re.IGNORECASE)


@dataclass
class RouteResult:
    type: str        # "jql" | "general"
    answer: str = "" # populated for type="general"

    @property
    def is_jql(self) -> bool:
        return self.type == "jql"


class QueryRouter:
    """Classifies a user query as JQL or general via a fast LLM call.

    Args:
        llm_client:  Any client with an async generate_jql(prompt) -> str method.
        prompt_file: Path to the router prompt template. Must contain {query}.
        two_pass:    When True (Ollama), a second LLM call generates the general
                     answer after classification. When False (Groq), the router
                     prompt produces the answer directly.
    """

    def __init__(self, llm_client, prompt_file: Path, two_pass: bool = False) -> None:
        self._llm_client = llm_client
        self._prompt_template = prompt_file.read_text(encoding="utf-8")
        self._two_pass = two_pass
        logger.info("QueryRouter loaded prompt from %s (two_pass=%s)", prompt_file, two_pass)

    def _check_override(self, query: str) -> tuple[str | None, str]:
        """Return (forced_type, clean_query). forced_type is 'jql', 'general', or None."""
        if _OVERRIDE_JQL_RE.search(query):
            return "jql", _OVERRIDE_JQL_RE.sub("", query).strip()
        if _OVERRIDE_GENERAL_RE.search(query):
            return "general", _OVERRIDE_GENERAL_RE.sub("", query).strip()
        return None, query

    async def route(self, query: str) -> RouteResult:
        """Classify query and return RouteResult."""
        forced_type, clean_query = self._check_override(query)

        if forced_type == "jql":
            logger.info("QueryRouter: user override → JQL")
            return RouteResult(type="jql")

        prompt = self._prompt_template.format(query=clean_query)
        logger.info("QueryRouter: classifying query")

        response = await self._llm_client.generate_jql(prompt)
        response = response.strip()

        if _JQL_SIGNAL_RE.match(response) and not _GENERAL_SIGNAL_RE.match(response):
            logger.info("QueryRouter: routed to JQL pipeline")
            return RouteResult(type="jql")

        logger.info("QueryRouter: routed to general answer (skipping RAG)")

        if forced_type == "general" or self._two_pass:
            answer_prompt = f"Answer the following question briefly and accurately:\n\n{clean_query}"
            answer = await self._llm_client.generate_jql(answer_prompt)
            return RouteResult(type="general", answer=answer.strip())

        return RouteResult(type="general", answer=response)
