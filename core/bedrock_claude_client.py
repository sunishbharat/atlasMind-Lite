import asyncio
import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from settings import (
    BEDROCK_API_KEY, CUSTOM_ENDPOINT, BEDROCK_REGION,
    BEDROCK_MODEL, BEDROCK_TEMPERATURE, BEDROCK_TIMEOUT, BEDROCK_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

# Marks the boundary between the stable system prompt and the dynamic RAG context.
# Everything before this marker is sent as the cached `system` block; everything
# after (field descriptions, JQL examples, user query) is the dynamic user message.
# Router and general-answer prompts don't contain this marker, so they are sent
# as a single user message without caching.
_CACHE_SPLIT_MARKER = "\n\n## Available Jira Fields"


class BedrockUnavailable(Exception):
    pass


class BedrockClaudeClient:
    def __init__(self, model: str = BEDROCK_MODEL):
        self.model = model
        self.api_key = BEDROCK_API_KEY
        self.temperature = BEDROCK_TEMPERATURE
        self.timeout = BEDROCK_TIMEOUT
        self.max_tokens = BEDROCK_MAX_TOKENS
        self._endpoint = CUSTOM_ENDPOINT
        if self.api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
        logger.info("Bedrock client initialized — model: %s  endpoint: %s", self.model, self._endpoint)

    def _boto_client(self):
        return boto3.client(
            service_name="bedrock-runtime",
            endpoint_url=self._endpoint,
            region_name=BEDROCK_REGION,
        )

    def test_connection(self) -> None:
        if not self.api_key:
            raise BedrockUnavailable("AWS_BEARER_TOKEN_BEDROCK is not set")
        logger.info("Bedrock client ready (API key present)")

    async def generate_jql(self, prompt: str) -> str:
        if not self.api_key:
            raise BedrockUnavailable("AWS_BEARER_TOKEN_BEDROCK is not set")

        logger.info("Bedrock generating response using model: %s", self.model)

        split_idx = prompt.find(_CACHE_SPLIT_MARKER)
        if split_idx != -1:
            system_text = prompt[:split_idx]
            user_text = prompt[split_idx:]
            logger.debug("Bedrock prompt caching active — system: %d chars, user: %d chars",
                         len(system_text), len(user_text))
        else:
            system_text = None
            user_text = prompt

        def _call() -> str:
            client = self._boto_client()
            kwargs: dict = dict(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": user_text}]}],
                inferenceConfig={
                    "temperature": self.temperature,
                    "maxTokens": self.max_tokens,
                },
            )
            if system_text:
                kwargs["system"] = [{"text": system_text, "cacheConfig": {"type": "default"}}]
            response = client.converse(**kwargs)
            return response["output"]["message"]["content"][0]["text"]

        try:
            text = await asyncio.to_thread(_call)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            raise BedrockUnavailable(f"Bedrock API error {code}: {msg}") from e
        except BotoCoreError as e:
            raise BedrockUnavailable(f"Bedrock connection error: {e}") from e

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
