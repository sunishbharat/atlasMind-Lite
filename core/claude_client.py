import logging

import anthropic

from settings import (
    CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_TIMEOUT, CLAUDE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class ClaudeUnavailable(Exception):
    pass


class ClaudeClient:
    def __init__(self, model: str = CLAUDE_MODEL):
        self.model = model
        self.api_key = CLAUDE_API_KEY
        self.temperature = CLAUDE_TEMPERATURE
        self.timeout = CLAUDE_TIMEOUT
        self.max_tokens = CLAUDE_MAX_TOKENS
        logger.info("Claude client initialized with model: %s", self.model)

    def test_connection(self) -> None:
        if not self.api_key:
            raise ClaudeUnavailable("CLAUDE_API_KEY is not set")
        logger.info("Claude client ready (API key present)")

    async def generate_jql(self, prompt: str) -> str:
        if not self.api_key:
            raise ClaudeUnavailable("CLAUDE_API_KEY is not set")

        logger.info("Claude client generating response using model: %s", self.model)
        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIConnectionError as e:
            raise ClaudeUnavailable("Anthropic API is not reachable") from e
        except anthropic.APITimeoutError as e:
            raise ClaudeUnavailable(f"Anthropic API timed out after {self.timeout}s") from e
        except anthropic.APIStatusError as e:
            raise ClaudeUnavailable(f"Anthropic API error {e.status_code}: {e.message}") from e

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
