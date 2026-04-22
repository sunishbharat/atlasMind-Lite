import httpx
import logging
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from settings import (
    VLLM_URL, VLLM_MODEL, VLLM_TEMPERATURE, VLLM_TIMEOUT,
    VLLM_MAX_TOKENS, VLLM_API_KEY,
)

logger = logging.getLogger(__name__)


class VllmUnavailable(Exception):
    pass


class VllmClient:
    def __init__(self, model: str = VLLM_MODEL):
        self.base_url = VLLM_URL.rstrip("/")
        self.model = model
        self.temperature = VLLM_TEMPERATURE
        self.timeout = VLLM_TIMEOUT
        self.max_tokens = VLLM_MAX_TOKENS
        self.api_key = VLLM_API_KEY
        logger.info(
            "vLLM client initialized — url: %s  model: %s",
            self.base_url,
            self.model or "(auto-detect)",
        )

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def test_connection(self) -> None:
        """Verify the vLLM server is reachable. Auto-detects the loaded model if VLLM_MODEL is unset."""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
        except RequestsConnectionError as e:
            raise VllmUnavailable(f"vLLM server not reachable at {self.base_url}") from e

        models = resp.json().get("data", [])
        if not models:
            raise VllmUnavailable("vLLM server is running but no model is loaded")

        if not self.model:
            self.model = models[0]["id"]
            logger.info("vLLM: auto-detected model %s", self.model)

    async def generate_jql(self, prompt: str) -> str:
        if not self.model:
            raise VllmUnavailable(
                "VLLM_MODEL is not set and auto-detection has not run — "
                "call test_connection() first or set VLLM_MODEL env var"
            )

        timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0)
        logger.info("vLLM generating response using model: %s", self.model)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.ReadTimeout as e:
            raise VllmUnavailable(
                f"vLLM read timed out after {self.timeout}s — "
                "increase VLLM_TIMEOUT or use a smaller/faster model"
            ) from e
        except httpx.ConnectError as e:
            raise VllmUnavailable(f"vLLM server not reachable at {self.base_url}") from e
        except httpx.HTTPStatusError as e:
            raise VllmUnavailable(
                f"vLLM error {e.response.status_code}: {e.response.text}"
            ) from e

        text = response.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()
