from pydantic import BaseModel, Field
from typing import Literal, Any

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    web_consent: bool = False

class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str = "spotify_web_app_architecture.pdf"
    page: int | None = None

class GraphFact(BaseModel):
    subject: str
    relation: str
    object: str
    entity_types: list[str] = []

class ChatResponse(BaseModel):
    answer: str
    status: Literal["answered", "needs_web_consent", "web_answer", "not_found", "error"]
    session_id: str
    sources: list[dict[str, Any]] = []
    graph_facts: list[dict[str, Any]] = []
    retrieval_validation: dict[str, Any] = {}
    web_used: bool = False
