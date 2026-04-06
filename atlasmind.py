import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from document_processor import DocumentProcessor
from jira_field_embeddings import Jira_Field_Embeddings
from jql_embeddings import JQL_Embeddings
from ollama_client import OllamaClient
from dconfig import EmbeddingsConfig
from settings import DEFAULT_ANNOTATION_FILE, DEFAULT_JIRA_FIELDS_FILE

logger = logging.getLogger(__name__)


class JqlResponse(BaseModel):
    jql: str | None
    chart_spec: dict[str, Any] | None
    answer: str


class AtlasMind:
    def __init__(self, embedconfig: EmbeddingsConfig):
        self.embedconfig = embedconfig
        self.ollama_client = OllamaClient()
        self.document_processor = DocumentProcessor(embedconfig=embedconfig)
        self.system_prompt_dir = Path("config/system_prompt.md")

        # Both embedding classes share the same DocumentProcessor so the
        # SentenceTransformer model is only loaded once.
        self.jql_embeddings = JQL_Embeddings(embedconfig, self.document_processor)
        self.jira_field_embeddings = Jira_Field_Embeddings(embedconfig, self.document_processor)

    def run(self):
        self.ollama_client.test_connection()
        self.jql_embeddings.run(Path(DEFAULT_ANNOTATION_FILE))
        self.jira_field_embeddings.run(Path(DEFAULT_JIRA_FIELDS_FILE))

    async def _build_prompt(self, query: str) -> str:
        """Build a RAG-grounded prompt combining system instructions with retrieved context.

        Prepends the system prompt (which defines the JSON output contract and
        mode-switching logic) with retrieved Jira field vocabulary and semantically
        similar JQL examples. The context section only provides data — it does not
        restate output format rules already covered by the system prompt.

        Args:
            query: The user's natural language query string.

        Returns:
            str: system_prompt + RAG context + user request, ready to send to Ollama.
        """
        model = self.document_processor._model

        # search_sample_jql_embeddings_db is synchronous; search_jira_fields is async
        jql_examples, _ = self.jql_embeddings.search_sample_jql_embeddings_db(query, model)
        jira_fields, _ = await self.jira_field_embeddings.search_jira_fields(query, model)

        system_prompt = self.system_prompt_dir.read_text(encoding="utf-8")

        # rows: (id, field_id, field_name, field_type, is_custom, description, distance)
        # description (row[5]) is built by _build_description and already includes
        # allowed values, JQL clause names, and field type — use it directly.
        fields_block = "\n".join(
            f"  - {row[5]}"
            for row in jira_fields
        )

        # rows: (id, annotation, jql, distance)
        examples_block = "\n\n".join(
            f"  -- {row[1]}\n  {row[2]}"
            for row in jql_examples
        )

        context = (
            "\n\n"
            "## Available Jira Fields\n"
            "Use only these field IDs when building JQL — do not invent fields.\n"
            f"{fields_block}\n\n"
            "## JQL Rules\n"
            "1. Use only field IDs and allowed values listed above — do not invent fields or values.\n"
            "2. Do not use placeholder values like 'ProjectName' or 'USERNAME'.\n"
            "3. If no specific project is mentioned, omit the project filter.\n"
            "4. Do NOT use date arithmetic between two fields — JQL does not support it.\n"
            "   INVALID: resolutiondate >= created + 20d\n"
            "   INVALID: resolutiondate - created > 20d\n"
            "   CORRECT: resolution IS NOT EMPTY ORDER BY resolutiondate DESC\n"
            "   (duration filtering is handled externally — omit it from JQL)\n"
            "5. Do NOT append LIMIT — result count is controlled externally.\n"
            "6. Always end with ORDER BY unless the user specifies otherwise.\n\n"
            "## Similar JQL Examples\n"
            f"{examples_block}\n\n"
            "## User Request\n"
            f"{query}\n"
        )

        return system_prompt + context


    async def generate_jql(self, query: str) -> JqlResponse:
        """Generate a JQL query (or general answer) from a natural language request.

        Builds a RAG-grounded prompt, sends it to Ollama, and parses the JSON
        response defined by the system prompt contract.

        Args:
            query: The user's natural language query string.

        Returns:
            JqlResponse with jql (None for general answers), chart_spec, and answer.

        Raises:
            ValueError: If Ollama returns a response that cannot be parsed as JSON.
        """
        prompt = await self._build_prompt(query)
        raw = await self.ollama_client.generate_jql(prompt)
        logger.debug("Ollama raw response: %s", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama response is not valid JSON: {raw!r}") from exc

        return JqlResponse(**data)