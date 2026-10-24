"""LangChain retrievers for the two halves of hybrid retrieval: FAISS (semantic) and Neo4j (graph)."""
import logging, re, threading, time
from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from .config import Settings

log = logging.getLogger(__name__)

STOPWORDS = {"what","which","where","when","does","have","with","from","that","this","about","tell","explain","please",
             "how","the","and","are","for","who","why","can","you","its","into","there","their","used","use"}

def query_terms(text: str) -> list[str]:
    """Meaningful lower-case terms (3+ chars, no stopwords), in order and de-duplicated."""
    terms = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
    return list(dict.fromkeys(t for t in terms if t not in STOPWORDS))

class FaissScoredRetriever(BaseRetriever):
    """Similarity search over the LangChain FAISS store that keeps the cosine score and rank in metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Any = None
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.vector_store is None: return []
        results = self.vector_store.similarity_search_with_score(query, k=self.k)
        docs = []
        for rank, (doc, inner_product) in enumerate(results, start=1):
            # Normalised inner product is cosine in [-1, 1]; map to [0, 1] for display.
            score = max(0.0, min(1.0, (float(inner_product) + 1.0) / 2.0))
            docs.append(Document(id=doc.id, page_content=doc.page_content,
                                 metadata={**doc.metadata, "score": score, "rank": rank, "retrieval": "FAISS"}))
        return docs

GRAPH_QUERY = """
MATCH (s)-[r]->(o)
WITH s, r, o, [term IN $terms WHERE toLower(coalesce(s.name,'')) CONTAINS term
    OR toLower(coalesce(o.name,'')) CONTAINS term
    OR toLower(coalesce(r.label,'')) CONTAINS term
    OR any(alias IN coalesce(s.aliases,[]) WHERE toLower(alias) CONTAINS term)
    OR any(alias IN coalesce(o.aliases,[]) WHERE toLower(alias) CONTAINS term)] AS hits
WHERE size(hits) > 0
RETURN s.name AS subject, labels(s) AS subject_types, coalesce(r.label,type(r)) AS relation,
       o.name AS object, labels(o) AS object_types, coalesce(r.source_text,'') AS source_text,
       coalesce(r.source_page, s.source_page, o.source_page) AS page, r.chunk_id AS chunk_id, size(hits) AS hits
ORDER BY hits DESC, page ASC
LIMIT $limit
"""

# Older graphs lack optional properties (aliases, chunk_id); suppress the per-query "property key does not exist" notices.
DRIVER_CONFIG = {"connection_timeout": 5, "notifications_min_severity": "OFF"}

class Neo4jConnection:
    """Lazily opens a langchain_neo4j.Neo4jGraph and retries after a cool-down, so an unreachable AuraDB never blocks startup."""
    RETRY_SECONDS = 60

    def __init__(self, cfg: Settings):
        self.cfg = cfg; self._graph = None; self._failed_at = 0.0; self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.cfg.neo4j_uri and self.cfg.neo4j_password)

    def get(self):
        if not self.configured: return None
        with self._lock:
            if self._graph is None and time.monotonic() - self._failed_at > self.RETRY_SECONDS:
                try:
                    from langchain_neo4j import Neo4jGraph
                    # refresh_schema=False: no APOC schema scan; the retriever runs a fixed, parameterised query.
                    self._graph = Neo4jGraph(url=self.cfg.neo4j_uri, username=self.cfg.neo4j_username, password=self.cfg.neo4j_password,
                                             database=self.cfg.neo4j_database, timeout=self.cfg.request_timeout_seconds,
                                             refresh_schema=False, driver_config=DRIVER_CONFIG)
                except Exception as exc:
                    self._failed_at = time.monotonic()
                    cause = exc.__cause__ or exc
                    log.warning("Unable to connect to Neo4j (%s: %s); graph retrieval disabled for %ss",
                                type(cause).__name__, str(cause)[:300], self.RETRY_SECONDS)
            return self._graph

    def close(self):
        if self._graph is not None:
            try: self._graph.close()
            except Exception: log.debug("Neo4j close failed", exc_info=True)

class Neo4jGraphRetriever(BaseRetriever):
    """Keyword-anchored graph retrieval of entity relationships; each fact links back to its source chunk."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: Any = None
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        terms = query_terms(query)
        graph = self.connection.get() if self.connection is not None else None
        if not terms or graph is None: return []
        try:
            rows = graph.query(GRAPH_QUERY, params={"terms": terms, "limit": self.k})
        except Exception:
            # Degrade to vector-only retrieval rather than failing the request.
            log.exception("Neo4j graph search failed")
            return []
        docs = []
        for rank, r in enumerate(rows, start=1):
            meta = {"subject": r["subject"], "subject_types": r.get("subject_types") or [], "relation": r["relation"],
                    "object": r["object"], "object_types": r.get("object_types") or [], "source_text": r.get("source_text") or "",
                    "page": r.get("page"), "chunk_id": r.get("chunk_id"), "rank": rank, "retrieval": "Neo4j"}
            docs.append(Document(page_content=f"{r['subject']} --{r['relation']}--> {r['object']}", metadata=meta))
        return docs
