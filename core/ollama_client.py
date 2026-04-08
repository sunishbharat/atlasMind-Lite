import requests
from settings import (
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    OLLAMA_TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_THREAD, OLLAMA_NUM_BATCH, OLLAMA_TOP_P, OLLAMA_TOP_K, OLLAMA_REPEAT_PENALTY,
)
from requests.exceptions import ConnectionError, Timeout
import logging
import httpx

logger = logging.getLogger(__name__)

class OllamaUnavailable(Exception):
    pass

class OllamaClient:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.url = f"{OLLAMA_URL}/api/generate"
        self.model = model
        self.timeout = OLLAMA_TIMEOUT
        self.options = {
            "num_ctx":        OLLAMA_NUM_CTX,
            "num_predict":    OLLAMA_NUM_PREDICT,
            "num_thread":     OLLAMA_NUM_THREAD,
            "num_batch":      OLLAMA_NUM_BATCH,
            "temperature":    OLLAMA_TEMPERATURE,
            "top_p":          OLLAMA_TOP_P,
            "top_k":          OLLAMA_TOP_K,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
        }
        logger.info(f"Ollama client initialized with model: {self.model}")

    def test_connection(self, prompt: str = "Hello") -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except ConnectionError as e:
            # Ollama not running / port closed
            raise OllamaUnavailable("Ollama server is not running on localhost:11434") from e
        except Timeout as e:
            raise OllamaUnavailable("Ollama request timed out") from e

        return resp.json()["response"]


    async def generate_jql(self, prompt: str) -> str:
        # LLM inference can be slow — use a short connect timeout but a long
        # read timeout. OLLAMA_TIMEOUT controls the read budget (default 120s,
        # override via JQL_OLLAMA_TIMEOUT env var).
        timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=10.0, pool=5.0)
        logger.info(f"Ollama client generating response using model : {self.model}")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.url,
                    json={"model": self.model, "prompt": prompt, "stream": False, "options": self.options},
                )
                response.raise_for_status()
        except httpx.ReadTimeout as e:
            raise OllamaUnavailable(
                f"Ollama read timed out after {self.timeout}s — "
                "increase JQL_OLLAMA_TIMEOUT or use a smaller model"
            ) from e
        except httpx.ConnectError as e:
            raise OllamaUnavailable("Ollama server is not reachable") from e
        text = response.json()["response"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]   # drop opening fence line (```json)
            text = text.rsplit("```", 1)[0]   # drop closing fence
        return text.strip()
    