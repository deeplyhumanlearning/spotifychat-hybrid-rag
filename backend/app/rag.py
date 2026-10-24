"""Hybrid RAG pipeline composed with LangChain Runnables (LCEL).

    question ─┬─ FaissScoredRetriever  (LangChain FAISS + OpenAIEmbeddings) ─┐
              └─ Neo4jGraphRetriever   (langchain_neo4j.Neo4jGraph)         ─┴─ rrf_fusion_and_validation
                                                                                 │ sufficient?
                                    no → ask for web-search consent  ◄───────────┤
                                    yes → ChatPromptTemplate | ChatOpenAI | StrOutputParser

Every step is a Runnable, so with LANGSMITH_TRACING=true each request produces one LangSmith trace containing the
retriever runs (with documents), the fusion step, the rendered prompt, the model call (tokens/latency) and the answer.
"""
import asyncio, logging
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Awaitable, Callable
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableParallel, RunnablePassthrough
from .config import Settings
from .fusion import ChunkCatalog, fact_record, rrf_fuse, validate_context
from .prompts import CONSENT_QUESTION, LOCAL_ANSWER_PROMPT, WEB_ANSWER_PROMPT
from .retrievers import FaissScoredRetriever, Neo4jConnection, Neo4jGraphRetriever
from .vectorstore import all_documents, build_embeddings, load_vector_store
from .websearch import web_search

log = logging.getLogger(__name__)
HISTORY_TURNS = 8

class PipelineError(Exception):
    status_code = 500; detail = "An unexpected server error occurred. Please try again."
    def __init__(self, detail: str | None = None):
        super().__init__(detail or self.detail)
        if detail: self.detail = detail

class RetrievalTimeout(PipelineError): status_code = 504; detail = "Knowledge-base retrieval timed out. Please try again."
class RetrievalFailed(PipelineError): status_code = 502; detail = "Knowledge-base retrieval failed. Check service health and ingestion."
class LLMNotConfigured(PipelineError): status_code = 503; detail = "The OpenAI API key is not configured on the server."
class LLMTimeout(PipelineError): status_code = 504; detail = "The language model request timed out. Please try again."
class LLMFailed(PipelineError): status_code = 502; detail = "The language model service returned an error. Please try again later."

@dataclass
class ChatOutcome:
    answer: str | None
    needs_consent: bool
    chunks: list[dict] = field(default_factory=list)
    graph_facts: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)

@dataclass
class WebOutcome:
    answer: str
    results: list[dict] = field(default_factory=list)

def format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(f"[Local source: {c.get('source') or 'PDF'}, page {c.get('page','?')}, chunk {c.get('chunk_id')} ]\n{c['text']}" for c in chunks) or "No local passages retrieved."

def format_graph(facts: list[dict]) -> str:
    return "\n".join(f"{g['subject']} --{g['relation']}--> {g['object']} (source page {g.get('page')})" for g in facts) or "No graph facts retrieved."

def format_web(results: list[dict]) -> str:
    return "\n".join(f"{w['title']} — {w['snippet']} ({w['url']})" for w in results) or "No results."

def to_messages(history: list[dict]):
    return convert_to_messages(history[-HISTORY_TURNS:])

class HybridRAGPipeline:
    def __init__(self, cfg: Settings, vector_store: Any, graph_connection: Neo4jConnection | None, llm: BaseChatModel | None,
                 vector_format: str = "langchain", search_fn: Callable[[str], Awaitable[list[dict]]] | None = None):
        self.cfg = cfg; self.vector_store = vector_store; self.vector_format = vector_format
        self.graph_connection = graph_connection; self.llm = llm
        self.search_fn = search_fn or (lambda q: web_search(q, cfg))
        self.catalog = ChunkCatalog(all_documents(vector_store))
        k = cfg.retrieval_top_k
        vector = FaissScoredRetriever(vector_store=vector_store, k=k).with_config(run_name="faiss_vector_retriever")
        graph = Neo4jGraphRetriever(connection=graph_connection, k=k).with_config(run_name="neo4j_graph_retriever")
        question = itemgetter("question")
        self.retrieval_chain = (
            RunnableParallel(question=question, vector=question | vector, graph=question | graph)
            | RunnableLambda(self._fuse, name="rrf_fusion_and_validation")
        ).with_config(run_name="hybrid_retrieval")
        self.answer_chain = self.web_answer_chain = None
        if llm is not None:
            self.answer_chain = (
                RunnablePassthrough.assign(context=lambda x: format_chunks(x["chunks"]), graph_context=lambda x: format_graph(x["graph_facts"]),
                                           history=lambda x: to_messages(x["history"]))
                | LOCAL_ANSWER_PROMPT | llm | StrOutputParser()
            ).with_config(run_name="grounded_answer")
            self.web_answer_chain = (
                RunnablePassthrough.assign(web_context=lambda x: format_web(x["web_results"]), history=lambda x: to_messages(x["history"]))
                | WEB_ANSWER_PROMPT | llm | StrOutputParser()
            ).with_config(run_name="web_answer")
        self.web_search = RunnableLambda(self._search, name="web_search")
        self.chat_flow = RunnableLambda(self._chat_flow, name="spotifychat_chat")
        self.web_flow = RunnableLambda(self._web_flow, name="spotifychat_web_answer")

    @property
    def vector_loaded(self) -> bool: return self.vector_store is not None

    @property
    def graph_configured(self) -> bool: return bool(self.graph_connection and self.graph_connection.configured)

    def _fuse(self, x: dict) -> dict:
        chunks = rrf_fuse(x["vector"], x["graph"], self.catalog, k=self.cfg.rrf_k, top_k=self.cfg.retrieval_top_k)
        facts = [fact_record(d) for d in x["graph"]][: self.cfg.retrieval_top_k]
        return {"chunks": chunks, "graph_facts": facts, "validation": validate_context(x["question"], chunks, facts)}

    async def _search(self, query: str) -> list[dict]:
        return await self.search_fn(query)

    async def _generate(self, chain, inputs: dict, config: RunnableConfig) -> str:
        if chain is None: raise LLMNotConfigured()
        try:
            answer = await chain.ainvoke(inputs, config)
        except openai.APITimeoutError as exc: raise LLMTimeout() from exc
        except openai.APIError as exc:
            log.exception("OpenAI API error"); raise LLMFailed() from exc
        return answer or "I could not generate an answer from the available context."

    async def _chat_flow(self, inputs: dict, config: RunnableConfig) -> ChatOutcome:
        try:
            retrieved = await asyncio.wait_for(self.retrieval_chain.ainvoke({"question": inputs["question"]}, config),
                                               timeout=self.cfg.request_timeout_seconds)
        except asyncio.TimeoutError as exc: raise RetrievalTimeout() from exc
        except Exception as exc:
            log.exception("Retrieval failed"); raise RetrievalFailed() from exc
        outcome = ChatOutcome(answer=None, needs_consent=True, chunks=retrieved["chunks"], graph_facts=retrieved["graph_facts"], validation=retrieved["validation"])
        if not outcome.validation["local_context_sufficient"]:
            return outcome  # no model call and no search: the API asks the user for consent
        outcome.answer = await self._generate(self.answer_chain, {"question": inputs["question"], "history": inputs["history"],
                                                                  "chunks": outcome.chunks, "graph_facts": outcome.graph_facts}, config)
        # The model may itself judge the local context insufficient; searching is still deferred to explicit consent.
        outcome.needs_consent = CONSENT_QUESTION.lower() in outcome.answer.lower()
        return outcome

    async def _web_flow(self, inputs: dict, config: RunnableConfig) -> WebOutcome:
        if self.web_answer_chain is None: raise LLMNotConfigured()
        try:
            results = await self.web_search.ainvoke(inputs["question"], config)
        except Exception:
            log.exception("Web search failed"); results = []
        answer = await self._generate(self.web_answer_chain, {"question": inputs["question"], "history": inputs["history"], "web_results": results}, config)
        return WebOutcome(answer=answer, results=results)

    async def achat(self, question: str, history: list[dict], session_id: str) -> ChatOutcome:
        return await self.chat_flow.ainvoke({"question": question, "history": history},
                                            {"metadata": {"session_id": session_id}, "tags": ["chat"]})

    async def aweb_answer(self, question: str, history: list[dict], session_id: str) -> WebOutcome:
        return await self.web_flow.ainvoke({"question": question, "history": history},
                                           {"metadata": {"session_id": session_id}, "tags": ["web-search", "user-consented"]})

    def close(self):
        if self.graph_connection: self.graph_connection.close()

def build_chat_model(cfg: Settings) -> BaseChatModel | None:
    if not cfg.openai_api_key: return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg.openai_model, api_key=cfg.openai_api_key, temperature=0.1, max_tokens=900,
                      timeout=cfg.request_timeout_seconds, max_retries=2)

def build_pipeline(cfg: Settings) -> HybridRAGPipeline:
    store, fmt = load_vector_store(cfg, build_embeddings(cfg))
    return HybridRAGPipeline(cfg, store, Neo4jConnection(cfg), build_chat_model(cfg), vector_format=fmt)
