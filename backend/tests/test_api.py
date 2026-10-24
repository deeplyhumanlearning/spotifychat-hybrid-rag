"""FastAPI contract: answering, web-search consent flow, session isolation, and error mapping."""
import pytest
from fastapi.testclient import TestClient
import app.main as main
from app.prompts import CONSENT_QUESTION
from app.rag import HybridRAGPipeline
from app.vectorstore import build_vector_store
from conftest import FakeConnection, FakeGraph, RecordingChatModel

REDIS_Q = "Where is playback state cached?"
OFFTOPIC_Q = "Who is the prime minister of India?"

class Harness:
    def __init__(self, cfg, embeddings, chunks):
        self.llm = RecordingChatModel(replies=[]); self.searches = []; self.search_error = None
        self.results = [{"title": "Result", "snippet": "Snippet", "url": "https://example.com"}]
        async def search(q):
            self.searches.append(q)
            if self.search_error: raise self.search_error
            return self.results
        self.graph = FakeGraph(rows=[{"subject": "Redis", "subject_types": ["Database"], "relation": "MENTIONED_WITH", "object": "PostgreSQL",
                                       "object_types": ["Database"], "source_text": "", "page": 2, "chunk_id": "chunk1"}])
        self.pipeline = HybridRAGPipeline(cfg, build_vector_store(chunks, embeddings), FakeConnection(self.graph), self.llm, search_fn=search)

@pytest.fixture
def h(cfg, embeddings, sample_chunks, monkeypatch):
    harness = Harness(cfg, embeddings, sample_chunks)
    monkeypatch.setattr(main, "build_pipeline", lambda _cfg: harness.pipeline)
    main.sessions.clear()
    with TestClient(main.app) as client:
        harness.client = client
        harness.chat = lambda sid, msg: client.post("/chat", json={"session_id": sid, "message": msg})
        yield harness

def test_health_reports_components_without_secrets(h):
    body = h.client.get("/health").json()
    assert body["vector_index_loaded"] is True and body["langsmith_tracing"] is False and body["neo4j_configured"] is True
    assert not any("key" in k.lower() for k in body) and "sk-" not in str(body)

def test_answer_from_local_context(h):
    h.llm.replies.append("Redis caches playback state (Source: page 2).")
    body = h.chat("session-aaaa", REDIS_Q).json()
    assert body["status"] == "answered" and body["answer"].startswith("Redis caches")
    assert {"chunk_id", "text", "page", "source", "rrf_score", "retrieval"} <= body["sources"][0].keys()
    assert body["graph_facts"][0]["relation"] == "MENTIONED_WITH" and body["retrieval_validation"]["fusion"] == "Reciprocal Rank Fusion"
    assert h.searches == []

def test_offtopic_question_requires_consent_and_does_not_search(h):
    body = h.chat("session-aaaa", OFFTOPIC_Q).json()
    assert body["status"] == "needs_web_consent" and body["answer"].endswith(CONSENT_QUESTION)
    assert h.searches == [] and h.llm.calls == []
    assert main.sessions["session-aaaa"]["pending_query"] == OFFTOPIC_Q

def test_yes_searches_the_original_pending_question(h):
    h.chat("session-aaaa", OFFTOPIC_Q)
    h.llm.replies.append("From the web: ...")
    body = h.chat("session-aaaa", "Yes please").json()
    assert h.searches == [OFFTOPIC_Q]  # the original question, not "Yes please"
    assert body["status"] == "web_answer" and body["web_used"] is True and body["sources"][0]["url"] == "https://example.com"
    assert OFFTOPIC_Q in h.llm.calls[0][-1].content and main.sessions["session-aaaa"]["pending_query"] is None
    # consent is single-use: another "yes" is not a search
    h.chat("session-aaaa", "yes"); assert h.searches == [OFFTOPIC_Q]

def test_no_declines_without_searching(h):
    h.chat("session-aaaa", OFFTOPIC_Q)
    body = h.chat("session-aaaa", "No thanks").json()
    assert body["status"] == "not_found" and "will not perform a web search" in body["answer"]
    assert h.searches == [] and main.sessions["session-aaaa"]["pending_query"] is None

def test_new_question_cancels_pending_offer(h):
    h.chat("session-aaaa", OFFTOPIC_Q)
    h.llm.replies.append("Redis.")
    assert h.chat("session-aaaa", REDIS_Q).json()["status"] == "answered"
    h.chat("session-aaaa", "yes"); assert h.searches == []

def test_consent_is_isolated_per_session(h):
    h.chat("session-aaaa", OFFTOPIC_Q)
    other = h.chat("session-bbbb", "yes").json()  # no pending offer in this session
    assert other["status"] == "needs_web_consent" and h.searches == []
    assert main.sessions["session-aaaa"]["pending_query"] == OFFTOPIC_Q
    h.llm.replies.append("web answer")
    h.chat("session-aaaa", "yes"); assert h.searches == [OFFTOPIC_Q]

def test_search_provider_failure_returns_not_found(h):
    h.search_error = RuntimeError("provider down"); h.chat("session-aaaa", OFFTOPIC_Q)
    h.llm.replies.append("Web search did not return usable results.")
    body = h.chat("session-aaaa", "yes").json()
    assert body["status"] == "not_found" and body["web_used"] is False

def test_openai_unavailable_maps_to_http_error(h):
    import httpx, openai
    h.llm.error = openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
    r = h.chat("session-aaaa", REDIS_Q)
    assert r.status_code == 504 and "timed out" in r.json()["detail"]

def test_openai_not_configured_maps_to_503(h):
    h.pipeline.answer_chain = None
    r = h.chat("session-aaaa", REDIS_Q)
    assert r.status_code == 503 and "not configured" in r.json()["detail"]

def test_neo4j_failure_still_answers(h):
    h.graph.error = RuntimeError("AuraDB unreachable"); h.llm.replies.append("Redis.")
    body = h.chat("session-aaaa", REDIS_Q).json()
    assert body["status"] == "answered" and body["graph_facts"] == []

def test_validation_errors_and_session_clear(h):
    assert h.chat("session-aaaa", " a ").status_code == 422
    assert h.client.post("/chat", json={"session_id": "short", "message": "hello"}).status_code == 422
    h.chat("session-aaaa", OFFTOPIC_Q)
    assert h.client.delete("/sessions/session-aaaa").json()["status"] == "cleared" and "session-aaaa" not in main.sessions
