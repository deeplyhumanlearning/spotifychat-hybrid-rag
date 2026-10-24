"""OpenAI embeddings and the FAISS vector store, both through LangChain integrations."""
import json, logging, warnings
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from .config import Settings

log = logging.getLogger(__name__)

# Vectors are L2-normalised and searched by inner product, i.e. cosine similarity (same as the original index).
STORE_KWARGS = {"distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT, "normalize_L2": True}
# LangChain warns that normalisation "is not applicable" to inner product, yet it does normalise both stored and query
# vectors, which is exactly what turns inner product into cosine. Silence that one misleading warning.
warnings.filterwarnings("ignore", message="Normalizing L2 is not applicable", category=UserWarning)

class UnconfiguredEmbeddings(Embeddings):
    """Placeholder so a saved index can still be loaded (and health reported) before OPENAI_API_KEY is set."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]: raise RuntimeError("OPENAI_API_KEY is not configured")
    def embed_query(self, text: str) -> list[float]: raise RuntimeError("OPENAI_API_KEY is not configured")

def build_embeddings(cfg: Settings) -> Embeddings:
    if not cfg.openai_api_key: return UnconfiguredEmbeddings()
    # Chunks are ~1k characters, far below the model's context, so skip client-side tiktoken re-chunking.
    return OpenAIEmbeddings(model=cfg.embedding_model, api_key=cfg.openai_api_key, request_timeout=cfg.request_timeout_seconds,
                            max_retries=2, check_embedding_ctx_length=False)

def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings, ids=[c.metadata["chunk_id"] for c in chunks], **STORE_KWARGS)

def save_vector_store(store: FAISS, cfg: Settings) -> None:
    Path(cfg.vector_store_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(cfg.vector_store_dir, index_name=cfg.vector_index_name)

def _load_legacy(cfg: Settings, embeddings: Embeddings) -> FAISS | None:
    """Wrap the pre-LangChain artifacts (raw IndexFlatIP + chunks.json) in a LangChain FAISS store without re-embedding."""
    index_path, chunks_path = Path(cfg.vector_index_path), Path(cfg.chunks_path)
    if not (index_path.exists() and chunks_path.exists()): return None
    import faiss
    index = faiss.read_index(str(index_path))
    rows = json.loads(chunks_path.read_text(encoding="utf-8"))
    if index.ntotal != len(rows): raise ValueError(f"Legacy index has {index.ntotal} vectors but {len(rows)} chunks")
    docs = {r["chunk_id"]: Document(id=r["chunk_id"], page_content=r["text"],
                                   metadata={"chunk_id": r["chunk_id"], "page": r.get("page"), "source": r.get("source", "")}) for r in rows}
    return FAISS(embeddings, index, InMemoryDocstore(docs), {i: r["chunk_id"] for i, r in enumerate(rows)}, **STORE_KWARGS)

def load_vector_store(cfg: Settings, embeddings: Embeddings) -> tuple[FAISS | None, str]:
    """Return (store, format). Prefers the LangChain-native store; falls back to legacy artifacts; never raises."""
    base = Path(cfg.vector_store_dir)
    try:
        if (base / f"{cfg.vector_index_name}.faiss").exists() and (base / f"{cfg.vector_index_name}.pkl").exists():
            # The .pkl docstore is produced by our own ingestion script inside the mounted index volume. Only load
            # index directories you created yourself: pickle deserialisation executes code from untrusted files.
            store = FAISS.load_local(str(base), embeddings, index_name=cfg.vector_index_name,
                                     allow_dangerous_deserialization=True, **STORE_KWARGS)
            return store, "langchain"
        store = _load_legacy(cfg, embeddings)
        if store is not None:
            log.warning("Loaded legacy FAISS index; re-run scripts/ingest.py to build the LangChain-native store.")
            return store, "legacy"
    except Exception:
        log.exception("Unable to load FAISS vector store")
    return None, "missing"

def all_documents(store: FAISS | None) -> list[Document]:
    """Every chunk in the store, in index order (used to resolve graph facts back to their source chunks)."""
    if store is None: return []
    return [store.docstore.search(doc_id) for doc_id in store.index_to_docstore_id.values()]
