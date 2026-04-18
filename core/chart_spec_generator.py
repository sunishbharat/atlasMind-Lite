"""
core/chart_spec_generator.py — focused LLM call to produce a ChartSpec.

Used by the /raw pipeline when the user appends a chart instruction after /raw.
The JQL is taken verbatim from the user; only the chart_spec is LLM-generated.

No RAG, no JQL context — a single lightweight prompt → JSON parse → ChartSpec.
"""

import json
import logging
from pathlib import Path

from core.models import ChartSpec

logger = logging.getLogger(__name__)


class ChartSpecGenerator:
    """Generates a validated ChartSpec from a natural-language chart instruction.

    Args:
        llm_client:  Any client with an async generate_jql(prompt) -> str method.
        prompt_file: Path to the chart spec prompt template. Must contain {chart_hint}.
    """

    def __init__(self, llm_client, prompt_file: Path) -> None:
        self._llm_client = llm_client
        self._prompt_template = prompt_file.read_text(encoding="utf-8")
        logger.info("ChartSpecGenerator loaded prompt from %s", prompt_file)

    async def generate(self, chart_hint: str) -> ChartSpec | None:
        """Generate a ChartSpec from a natural language chart instruction.

        Args:
            chart_hint: The user's chart instruction extracted from after /raw,
                e.g. "stacked bar chart with x=status and y=count by assignee".

        Returns:
            A validated ChartSpec instance, or None if the LLM output cannot
            be parsed or fails Pydantic validation.
        """
        prompt = self._prompt_template.replace("{chart_hint}", chart_hint)
        raw = await self._llm_client.generate_jql(prompt)
        raw = raw.strip()

        try:
            data = json.loads(raw, strict=False)
            spec = ChartSpec(**data)
            logger.info(
                "ChartSpecGenerator: type=%s x_field=%s y_field=%s color_field=%s",
                spec.type, spec.x_field, spec.y_field, spec.color_field,
            )
            return spec
        except json.JSONDecodeError as exc:
            logger.warning("ChartSpecGenerator: LLM returned invalid JSON %r — %s", raw, exc)
            return None
        except ValueError as exc:
            logger.warning("ChartSpecGenerator: ChartSpec validation failed %r — %s", raw, exc)
            return None
