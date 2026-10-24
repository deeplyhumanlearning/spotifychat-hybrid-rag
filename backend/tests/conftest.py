"""Shared fixtures. No test needs network access or real credentials: OpenAI, Neo4j and the search provider are faked."""
import os, sys
from pathlib import Path

# Isolate tests from any developer .env / shell secrets before the app modules read settings.
for var in ("OPENAI_API_KEY","NEO4J_URI","NEO4J_PASSWORD","SERPER_API_KEY","SERPAPI_API_KEY","LANGSMITH_API_KEY","LANGCHAIN_API_KEY","LANGCHAIN_TRACING_V2"):
    os.environ[var] = ""
os.environ["LANGSMITH_TRACING"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from app.config import Settings

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "source" / "spotify_web_app_architecture.pdf"

class RecordingChatModel(BaseChatModel):
    """Fake chat model: returns queued replies (or raises `error`) and records the messages it was sent."""
    replies: list[str] = []
    calls: list = []
    error: Exception | None = None

    @property
    def _llm_type(self) -> str: return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        if self.error is not None: raise self.error
        text = self.replies.pop(0) if self.replies else "ok"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

class FakeGraph:
    """Stands in for langchain_neo4j.Neo4jGraph."""
    def __init__(self, rows=None, error=None): self.rows = rows or []; self.error = error; self.queries = []
    def query(self, cypher, params=None):
        self.queries.append((cypher, params))
        if self.error: raise self.error
        return self.rows
    def close(self): pass

class FakeConnection:
    def __init__(self, graph): self.graph = graph; self.configured = graph is not None
    def get(self): return self.graph
    def close(self): pass

@pytest.fixture
def cfg(tmp_path):
    return Settings(_env_file=None, openai_api_key="", neo4j_uri="", neo4j_password="", vector_store_dir=str(tmp_path),
                    vector_index_path=str(tmp_path / "index.faiss"), chunks_path=str(tmp_path / "chunks.json"), retrieval_top_k=4)

@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=32)

@pytest.fixture
def sample_chunks():
    texts = [
        (1, "The Auth & Identity Service issues JWT access tokens and refresh tokens. OAuth2 / OIDC login is supported."),
        (2, "Playback state is cached in Redis. PostgreSQL stores users, playlists and subscriptions."),
        (3, "The Media Processor Service transcodes audio into HLS renditions stored in S3-compatible object storage."),
        (4, "Kafka carries analytics events from every service to ClickHouse for reporting."),
    ]
    docs = []
    for i, (page, text) in enumerate(texts):
        cid = f"chunk{i}"
        docs.append(Document(id=cid, page_content=text, metadata={"chunk_id": cid, "page": page, "source": "arch.pdf"}))
    return docs
