"""Neo4j graph retrieval, Reciprocal Rank Fusion, and context sufficiency validation."""
import pytest
from langchain_core.documents import Document
from app.fusion import ChunkCatalog, rrf_fuse, validate_context
from app.retrievers import Neo4jConnection, Neo4jGraphRetriever, query_terms
from conftest import FakeConnection, FakeGraph

def row(subject, obj, page, chunk_id=None, source_text=""):
    return {"subject": subject, "subject_types": ["ArchitectureEntity"], "relation": "MENTIONED_WITH", "object": obj,
            "object_types": ["ArchitectureEntity"], "source_text": source_text, "page": page, "chunk_id": chunk_id, "hits": 1}

def vec(cid, rank): return Document(id=cid, page_content=f"text {cid}", metadata={"chunk_id": cid, "page": 1, "score": 0.9, "rank": rank})

def fact(cid=None, source_text="", page=None):
    return Document(page_content="a --MENTIONED_WITH--> b", metadata={"subject": "a", "relation": "MENTIONED_WITH", "object": "b",
                                                                       "chunk_id": cid, "source_text": source_text, "page": page})

def test_query_terms_drop_stopwords_and_duplicates():
    assert query_terms("How does the Redis cache work with Redis?") == ["redis", "cache", "work"]

def test_graph_retriever_returns_documents_with_fact_metadata():
    graph = FakeGraph(rows=[row("Redis", "PostgreSQL", 2, "chunk1")])
    docs = Neo4jGraphRetriever(connection=FakeConnection(graph), k=5).invoke("Where is Redis used?")
    assert docs[0].page_content == "Redis --MENTIONED_WITH--> PostgreSQL"
    assert {"subject": "Redis", "object": "PostgreSQL", "page": 2, "chunk_id": "chunk1", "rank": 1, "retrieval": "Neo4j"}.items() <= docs[0].metadata.items()
    cypher, params = graph.queries[0]
    assert params == {"terms": ["redis"], "limit": 5} and "$terms" in cypher  # parameterised, never string-built

def test_graph_retriever_degrades_gracefully():
    assert Neo4jGraphRetriever(connection=FakeConnection(FakeGraph(error=RuntimeError("AuraDB down")))).invoke("redis cache") == []
    assert Neo4jGraphRetriever(connection=FakeConnection(None)).invoke("redis cache") == []
    assert Neo4jGraphRetriever(connection=FakeConnection(FakeGraph(rows=[row("a", "b", 1)]))).invoke("what is the") == []

def test_unconfigured_neo4j_connection_never_connects(cfg):
    conn = Neo4jConnection(cfg)
    assert conn.configured is False and conn.get() is None

def test_rrf_ranks_chunks_supported_by_both_retrievers_first(sample_chunks):
    catalog = ChunkCatalog(sample_chunks)
    vector = [vec("chunk0", 1), vec("chunk1", 2), vec("chunk2", 3)]
    graph = [fact("chunk2"), fact("chunk2"), fact("chunk1")]  # chunk2 best in graph; duplicate facts count once
    fused = rrf_fuse(vector, graph, catalog, k=60, top_k=10)
    assert [c["chunk_id"] for c in fused] == ["chunk2", "chunk1", "chunk0"]
    top = fused[0]
    assert top["rrf_score"] == pytest.approx(1 / 63 + 1 / 61) and top["retrieval"] == "FAISS+Neo4j" and top["vector_rank"] == 3 and top["graph_rank"] == 1
    assert fused[2]["retrieval"] == "FAISS" and fused[2]["rank"] == 3

def test_rrf_adds_graph_only_chunks_resolved_by_id_or_source_text(sample_chunks):
    catalog = ChunkCatalog(sample_chunks)
    legacy_fact = fact(source_text=sample_chunks[3].page_content[:800], page=4)  # pre-chunk_id graph edge
    fused = rrf_fuse([vec("chunk0", 1)], [fact("chunk2"), legacy_fact, fact("unknown")], catalog, top_k=10)
    by_id = {c["chunk_id"]: c for c in fused}
    assert set(by_id) == {"chunk0", "chunk2", "chunk3"}
    assert by_id["chunk3"]["retrieval"] == "Neo4j" and by_id["chunk3"]["text"] == sample_chunks[3].page_content
    assert "score" not in by_id["chunk3"]  # no cosine score for graph-only chunks; UI falls back to rrf_score

def test_rrf_truncates_to_top_k(sample_chunks):
    fused = rrf_fuse([vec(f"chunk{i}", i + 1) for i in range(4)], [], ChunkCatalog(sample_chunks), top_k=2)
    assert [c["chunk_id"] for c in fused] == ["chunk0", "chunk1"]

def test_validate_context_requires_chunks_and_term_overlap():
    chunks = [{"text": "Playback state is cached in Redis."}]
    ok = validate_context("Where is playback state cached?", chunks, [])
    assert ok["local_context_sufficient"] and ok["query_terms_matched"] == ["cached", "playback", "state"]
    assert not validate_context("Who is the prime minister of India?", chunks, [])["local_context_sufficient"]
    assert not validate_context("Where is playback state cached?", [], [{"subject": "Playback", "relation": "x", "object": "y"}])["local_context_sufficient"]
