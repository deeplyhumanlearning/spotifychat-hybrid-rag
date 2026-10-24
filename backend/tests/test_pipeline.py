"""The LCEL pipeline end to end with fakes: retrieval -> RRF -> validation -> prompt -> chat model, plus tracing structure."""
import asyncio
import httpx, openai, pytest
from langchain_core.tracers.context import collect_runs
from app.prompts import CONSENT_QUESTION, LOCAL_ANSWER_PROMPT
from app.rag import HybridRAGPipeline, LLMFailed, LLMNotConfigured, LLMTimeout, RetrievalFailed, RetrievalTimeout
from app.vectorstore import build_vector_store
from conftest import FakeConnection, FakeGraph, RecordingChatModel

REDIS_Q = "Where is playback state cached?"
OFFTOPIC_Q = "Who is the prime minister of India?"

def make_pipeline(cfg, embeddings, chunks, llm=None, graph=None, store=..., search_fn=None):
    store = build_vector_store(chunks, embeddings) if store is ... else store
    graph = graph if graph is not None else FakeGraph(rows=[{"subject": "Redis", "subject_types": [], "relation": "MENTIONED_WITH", "object": "PostgreSQL",
                                                             "object_types": [], "source_text": "", "page": 2, "chunk_id": "chunk1"}])
    return HybridRAGPipeline(cfg, store, FakeConnection(graph), llm, search_fn=search_fn)

def run(coro): return asyncio.run(coro)

def test_grounded_answer_uses_fused_context_in_langchain_prompt(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(replies=["Redis caches playback state (Source: page 2)."])
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm).achat(REDIS_Q, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "s1"))
    assert out.answer.startswith("Redis caches") and out.needs_consent is False
    assert out.validation["local_context_sufficient"] and out.graph_facts[0]["subject"] == "Redis"
    assert out.chunks[0]["chunk_id"] == "chunk1" and out.chunks[0]["retrieval"] == "FAISS+Neo4j"  # graph support lifts chunk1 to the top
    messages = llm.calls[0]
    assert [m.type for m in messages] == ["system", "human", "ai", "human"]  # system, history, question
    prompt = messages[-1].content
    assert REDIS_Q in prompt and "Playback state is cached in Redis" in prompt and "Redis --MENTIONED_WITH--> PostgreSQL" in prompt

def test_prompt_template_renders_expected_variables():
    msgs = LOCAL_ANSWER_PROMPT.invoke({"question": "Q?", "context": "CTX", "graph_context": "G", "history": []}).to_messages()
    assert "SpotifyChat" in msgs[0].content and "CTX" in msgs[-1].content and CONSENT_QUESTION in msgs[-1].content

def test_insufficient_context_asks_consent_without_model_or_search(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(); searched = []
    async def search(q): searched.append(q); return []
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm, search_fn=search).achat(OFFTOPIC_Q, [], "s1"))
    assert out.answer is None and out.needs_consent and not out.validation["local_context_sufficient"]
    assert llm.calls == [] and searched == []

def test_model_judged_insufficient_requests_consent(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(replies=["The document does not say. " + CONSENT_QUESTION])
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm).achat(REDIS_Q, [], "s1"))
    assert out.needs_consent and out.answer.endswith(CONSENT_QUESTION)

def test_web_answer_uses_search_results_through_langchain(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(replies=["Per example.com, ..."]); searched = []
    async def search(q): searched.append(q); return [{"title": "T", "snippet": "S", "url": "https://example.com"}]
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm, search_fn=search).aweb_answer(OFFTOPIC_Q, [], "s1"))
    assert searched == [OFFTOPIC_Q] and out.results[0]["url"] == "https://example.com"
    assert "T — S (https://example.com)" in llm.calls[0][-1].content and "Web search was explicitly approved" in llm.calls[0][0].content

def test_search_provider_failure_still_answers_without_results(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(replies=["Web search did not return usable results."])
    async def search(q): raise httpx.ConnectError("provider down")
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm, search_fn=search).aweb_answer(OFFTOPIC_Q, [], "s1"))
    assert out.results == [] and "No results." in llm.calls[0][-1].content

def test_neo4j_unavailable_falls_back_to_vector_only(cfg, embeddings, sample_chunks):
    llm = RecordingChatModel(replies=["answer"])
    out = run(make_pipeline(cfg, embeddings, sample_chunks, llm, graph=FakeGraph(error=RuntimeError("down"))).achat(REDIS_Q, [], "s1"))
    assert out.answer == "answer" and out.graph_facts == [] and all(c["retrieval"] == "FAISS" for c in out.chunks)

def test_missing_faiss_index_means_no_local_context(cfg, embeddings, sample_chunks):
    out = run(make_pipeline(cfg, embeddings, sample_chunks, RecordingChatModel(), store=None).achat(REDIS_Q, [], "s1"))
    assert out.answer is None and out.chunks == []

@pytest.mark.parametrize("error, expected", [
    (openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")), LLMTimeout),
    (openai.APIError("boom", request=httpx.Request("POST", "https://api.openai.com"), body=None), LLMFailed),
])
def test_openai_errors_map_to_pipeline_errors(cfg, embeddings, sample_chunks, error, expected):
    with pytest.raises(expected):
        run(make_pipeline(cfg, embeddings, sample_chunks, RecordingChatModel(error=error)).achat(REDIS_Q, [], "s1"))

def test_unconfigured_openai_is_reported(cfg, embeddings, sample_chunks):
    with pytest.raises(LLMNotConfigured):
        run(make_pipeline(cfg, embeddings, sample_chunks, llm=None).achat(REDIS_Q, [], "s1"))

def test_retrieval_errors_and_timeouts(cfg, embeddings, sample_chunks):
    class Broken:
        index_to_docstore_id = {}
        def similarity_search_with_score(self, *a, **k): raise RuntimeError("embedding service down")
    with pytest.raises(RetrievalFailed):
        run(make_pipeline(cfg, embeddings, sample_chunks, RecordingChatModel(), store=Broken()).achat(REDIS_Q, [], "s1"))
    class Slow(Broken):
        def similarity_search_with_score(self, *a, **k): import time; time.sleep(0.5); return []
    cfg.request_timeout_seconds = 0.05
    with pytest.raises(RetrievalTimeout):
        run(make_pipeline(cfg, embeddings, sample_chunks, RecordingChatModel(), store=Slow()).achat(REDIS_Q, [], "s1"))

def test_single_trace_contains_every_pipeline_stage(cfg, embeddings, sample_chunks):
    """What LangSmith receives: one root run per request with retrieval, fusion, prompt and model runs nested inside."""
    with collect_runs() as cb:
        run(make_pipeline(cfg, embeddings, sample_chunks, RecordingChatModel(replies=["ok"])).achat(REDIS_Q, [], "session-123"))
    assert len(cb.traced_runs) == 1
    root = cb.traced_runs[0]
    names = set()
    def walk(r): names.add(r.name); [walk(c) for c in r.child_runs]
    walk(root)
    assert root.name == "spotifychat_chat" and root.extra["metadata"]["session_id"] == "session-123"
    assert {"hybrid_retrieval", "faiss_vector_retriever", "neo4j_graph_retriever", "rrf_fusion_and_validation",
            "grounded_answer", "ChatPromptTemplate", "RecordingChatModel", "StrOutputParser"} <= names
