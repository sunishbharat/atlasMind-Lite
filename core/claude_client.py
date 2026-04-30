import logging

import anthropic

from settings import (
    CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_TIMEOUT, CLAUDE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Marks the boundary between the stable system prompt and the dynamic RAG context.
# Everything before this marker is sent as the cached system block; everything
# after (field descriptions, JQL examples, user query) is the dynamic user message.
# Router and general-answer prompts don't contain this marker, so they are sent
# as a single user message without caching.
_CACHE_SPLIT_MARKER = "\n\n## Available Jira Fields"


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

        split_idx = prompt.find(_CACHE_SPLIT_MARKER)
        if split_idx != -1:
            system_text = prompt[:split_idx]
            user_text = prompt[split_idx:]
            logger.debug(
                "Claude prompt caching active — system: %d chars, user: %d chars",
                len(system_text), len(user_text),
            )
            system_block = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
        else:
            system_block = None
            user_text = prompt

        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            kwargs: dict = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": user_text}],
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            )
            if system_block:
                kwargs["system"] = system_block
            response = await client.messages.create(**kwargs)
        except anthropic.APIConnectionError as e:
            raise ClaudeUnavailable("Anthropic API is not reachable") from e
        except anthropic.APITimeoutError as e:
            raise ClaudeUnavailable(f"Anthropic API timed out after {self.timeout}s") from e
        except anthropic.APIStatusError as e:
            raise ClaudeUnavailable(f"Anthropic API error {e.status_code}: {e.message}") from e

        usage = response.usage
        logger.info(
            "Claude token usage — input: %d, output: %d, cache_read: %d, cache_write: %d",
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            getattr(usage, "cache_creation_input_tokens", 0),
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
