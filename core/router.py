"""
core/router.py — fast query router.

QueryRouter makes a single lightweight LLM call to decide whether a query
needs the full RAG + JQL pipeline or can be answered directly.

The router prompt lives in config/router_prompt.md — edit that file to tune
routing behaviour without touching Python code.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_JQL_SIGNAL_RE = re.compile(r"^jql\b", re.IGNORECASE)


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
        llm_client: Any client with an async generate_jql(prompt) -> str method
                    (OllamaClient or GroqClient).
        prompt_file: Path to the router prompt template. Must contain {query}.
    """

    def __init__(self, llm_client, prompt_file: Path) -> None:
        self._llm_client = llm_client
        self._prompt_template = prompt_file.read_text(encoding="utf-8")
        logger.info("QueryRouter loaded prompt from %s", prompt_file)

    async def route(self, query: str) -> RouteResult:
        """Classify query and return RouteResult.

        Returns:
            RouteResult(type="jql")                  — caller should run RAG pipeline.
            RouteResult(type="general", answer="...") — caller can return immediately.
        """
        prompt = self._prompt_template.format(query=query)
        logger.info("QueryRouter: classifying query")

        response = await self._llm_client.generate_jql(prompt)
        response = response.strip()

        if _JQL_SIGNAL_RE.match(response):
            logger.info("QueryRouter: routed to JQL pipeline")
            return RouteResult(type="jql")

        logger.info("QueryRouter: routed to general answer (skipping RAG)")
        return RouteResult(type="general", answer=response)
