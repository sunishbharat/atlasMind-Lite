import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from atlasmind import AtlasMind, JqlResponse
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_atlasmind: AtlasMind | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _atlasmind
    logger.info("Starting up — seeding pgvector databases...")
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    _atlasmind = AtlasMind(config)
    _atlasmind.run()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="aMind JQL Generator", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=JqlResponse)
async def query(request: QueryRequest):
    try:
        return await _atlasmind.generate_jql(request.query)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
