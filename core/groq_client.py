import httpx
import logging
from settings import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_TIMEOUT, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqUnavailable(Exception):
    pass


class GroqClient:
    def __init__(self, model: str = GROQ_MODEL):
        self.model = model
        self.api_key = GROQ_API_KEY
        self.temperature = GROQ_TEMPERATURE
        self.timeout = GROQ_TIMEOUT
        self.max_tokens = GROQ_MAX_TOKENS
        logger.info("Groq client initialized with model: %s", self.model)

    def test_connection(self) -> None:
        if not self.api_key:
            raise GroqUnavailable("GROQ_API_KEY is not set")
        logger.info("Groq client ready (API key present)")

    async def generate_jql(self, prompt: str) -> str:
        if not self.api_key:
            raise GroqUnavailable("GROQ_API_KEY is not set")

        timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0)
        logger.info("Groq client generating response using model: %s", self.model)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    _GROQ_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
        except httpx.ReadTimeout as e:
            raise GroqUnavailable(
                f"Groq read timed out after {self.timeout}s"
            ) from e
        except httpx.ConnectError as e:
            raise GroqUnavailable("Groq API is not reachable") from e
        except httpx.HTTPStatusError as e:
            raise GroqUnavailable(f"Groq API error {e.response.status_code}: {e.response.text}") from e

        text = response.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
