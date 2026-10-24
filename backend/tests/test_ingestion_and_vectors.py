"""Document loading/chunking, LangChain embeddings configuration, and FAISS storage/retrieval."""
import json
import faiss, numpy as np, pytest
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from app.documents import chunk_id_for, load_pdf, split_documents
from app.retrievers import FaissScoredRetriever
from app.vectorstore import (UnconfiguredEmbeddings, all_documents, build_embeddings, build_vector_store, load_vector_store,
                             save_vector_store)
from conftest import PDF_PATH

def test_pdf_loads_as_langchain_documents_with_page_metadata():
    pages = load_pdf(PDF_PATH)
    assert pages and all(p.metadata["source"] == PDF_PATH.name for p in pages)
    assert [p.metadata["page"] for p in pages] == list(range(1, len(pages) + 1))  # 1-based pages, in order
    assert all("\x00" not in p.page_content for p in pages)

def test_split_documents_respects_size_and_assigns_stable_ids():
    chunks = split_documents(load_pdf(PDF_PATH), chunk_size=800, chunk_overlap=100)
    assert len(chunks) > 1 and all(len(c.page_content) <= 800 for c in chunks)
    for c in chunks:
        assert c.metadata["chunk_id"] == c.id == chunk_id_for(c.metadata["page"], c.page_content)
        assert {"source", "page", "start_index"} <= c.metadata.keys()
    assert len({c.id for c in chunks}) == len(chunks)

def test_build_embeddings_uses_langchain_openai(cfg):
    assert isinstance(build_embeddings(cfg), UnconfiguredEmbeddings)
    cfg.openai_api_key = "sk-test-not-real"
    emb = build_embeddings(cfg)
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == cfg.embedding_model and emb.check_embedding_ctx_length is False

def test_unconfigured_embeddings_fail_clearly():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        UnconfiguredEmbeddings().embed_query("hello")

def test_faiss_store_roundtrip_and_scored_retrieval(cfg, embeddings, sample_chunks):
    save_vector_store(build_vector_store(sample_chunks, embeddings), cfg)
    store, fmt = load_vector_store(cfg, embeddings)
    assert fmt == "langchain" and isinstance(store, FAISS) and len(all_documents(store)) == 4
    docs = FaissScoredRetriever(vector_store=store, k=3).invoke(sample_chunks[2].page_content)
    assert len(docs) == 3 and docs[0].metadata["chunk_id"] == "chunk2"
    assert docs[0].metadata["score"] == pytest.approx(1.0, abs=1e-4)  # identical text: cosine 1 -> score 1
    assert [d.metadata["rank"] for d in docs] == [1, 2, 3] and all(0 <= d.metadata["score"] <= 1 for d in docs)

def test_legacy_raw_faiss_index_is_wrapped_without_reembedding(cfg, embeddings, sample_chunks):
    vecs = np.asarray(embeddings.embed_documents([c.page_content for c in sample_chunks]), dtype="float32")
    faiss.normalize_L2(vecs); index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
    faiss.write_index(index, cfg.vector_index_path)
    rows = [{"chunk_id": c.id, "text": c.page_content, "page": c.metadata["page"], "source": "arch.pdf"} for c in sample_chunks]
    open(cfg.chunks_path, "w", encoding="utf-8").write(json.dumps(rows))
    store, fmt = load_vector_store(cfg, embeddings)
    assert fmt == "legacy"
    assert FaissScoredRetriever(vector_store=store, k=1).invoke(sample_chunks[1].page_content)[0].metadata["chunk_id"] == "chunk1"

def test_missing_or_corrupt_index_is_reported_not_raised(cfg, embeddings):
    assert load_vector_store(cfg, embeddings) == (None, "missing")
    open(cfg.vector_index_path, "wb").write(b"not a faiss index"); open(cfg.chunks_path, "w").write("[]")
    assert load_vector_store(cfg, embeddings) == (None, "missing")
    assert FaissScoredRetriever(vector_store=None).invoke("anything") == []
