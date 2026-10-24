"""Reciprocal Rank Fusion of FAISS and Neo4j results, plus the local-context sufficiency check."""
from typing import Any
from langchain_core.documents import Document
from .retrievers import query_terms

class ChunkCatalog:
    """Resolves a graph fact to the text chunk it was extracted from (by chunk_id, or by the stored source_text prefix)."""
    def __init__(self, chunks: list[Document]):
        self.chunks = [c for c in chunks if c is not None]
        self.by_id = {c.metadata.get("chunk_id"): c for c in self.chunks}
        self._flat = [_flat(c.page_content) for c in self.chunks]

    def resolve(self, fact: Document) -> Document | None:
        cid = fact.metadata.get("chunk_id")
        if cid and cid in self.by_id: return self.by_id[cid]
        # Graphs ingested before chunk_id was stored keep the first 800 chars of the source chunk on the edge. Match a
        # whitespace-insensitive excerpt anywhere in a chunk, so edges survive re-chunking of the same document.
        snippet = _flat(fact.metadata.get("source_text") or "")[:120]
        if len(snippet) < 20: return None
        page = fact.metadata.get("page")
        for c, flat in zip(self.chunks, self._flat):
            if (page is None or c.metadata.get("page") == page) and snippet in flat: return c
        return None

def _flat(text: str) -> str:
    return " ".join(text.split())

def chunk_record(doc: Document) -> dict[str, Any]:
    m = doc.metadata
    rec = {"chunk_id": m.get("chunk_id") or doc.id, "text": doc.page_content, "page": m.get("page"), "source": m.get("source", "")}
    if "score" in m: rec["score"] = m["score"]
    return rec

def fact_record(doc: Document) -> dict[str, Any]:
    return {k: doc.metadata.get(k) for k in ("subject","subject_types","relation","object","object_types","source_text","page","chunk_id","rank","retrieval")}

def rrf_fuse(vector_docs: list[Document], graph_docs: list[Document], catalog: ChunkCatalog, k: int = 60, top_k: int = 6) -> list[dict[str, Any]]:
    """Fuse two rankings of *chunks*: FAISS ranks chunks directly; the graph ranks the chunks its matching facts came from.

    score(chunk) = sum over lists of 1 / (k + rank). A chunk supported by both retrievers outranks one found by either alone,
    and a chunk reached only through the graph can enter the context even if FAISS missed it.
    """
    scores: dict[str, float] = {}; records: dict[str, dict] = {}; via: dict[str, set] = {}
    for rank, doc in enumerate(vector_docs, start=1):
        cid = doc.metadata.get("chunk_id") or doc.id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        records[cid] = {**chunk_record(doc), "vector_rank": rank}; via.setdefault(cid, set()).add("FAISS")
    graph_rank = 0; seen = set()
    for fact in graph_docs:
        chunk = catalog.resolve(fact)
        if chunk is None: continue
        cid = chunk.metadata.get("chunk_id") or chunk.id
        if cid in seen: continue  # rank each chunk once, at the position of its best-ranked fact
        seen.add(cid); graph_rank += 1
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + graph_rank)
        records.setdefault(cid, chunk_record(chunk))["graph_rank"] = graph_rank; via.setdefault(cid, set()).add("Neo4j")
    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
    return [{**records[c], "rrf_score": scores[c], "rank": i, "retrieval": "+".join(sorted(via[c]))} for i, c in enumerate(ranked, start=1)]

def validate_context(question: str, chunks: list[dict], graph_facts: list[dict]) -> dict[str, Any]:
    """Lightweight lexical relevance gate: at least one meaningful question term must occur in the fused context."""
    terms = query_terms(question)
    corpus = " ".join([c.get("text", "") for c in chunks] + [f"{g.get('subject','')} {g.get('relation','')} {g.get('object','')}" for g in graph_facts]).lower()
    overlap = sorted(t for t in terms if t in corpus)
    return {"local_context_sufficient": bool(chunks and overlap), "query_terms_matched": overlap, "vector_chunks": len(chunks),
            "graph_facts": len(graph_facts), "fusion": "Reciprocal Rank Fusion"}
