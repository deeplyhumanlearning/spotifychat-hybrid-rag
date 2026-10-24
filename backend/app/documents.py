"""Architecture-document loading and chunking with LangChain document loaders and text splitters."""
import hashlib, re
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_id_for(page: int, text: str) -> str:
    """Stable chunk id (same scheme as the original ingestion, so graph links and citations stay valid)."""
    return hashlib.sha1(f"{page}:{text}".encode()).hexdigest()[:16]

def _clean(text: str) -> str:
    # PDF extraction yields ragged whitespace and NULs; keep line breaks so the splitter can use them as boundaries.
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.replace("\x00", "").splitlines()]
    return "\n".join(line for line in lines if line)

def load_pdf(pdf_path: str | Path) -> list[Document]:
    """One Document per PDF page, with 1-based `page` and the file name as `source`."""
    pdf_path = Path(pdf_path)
    pages = []
    for doc in PyPDFLoader(str(pdf_path), mode="page").load():
        text = _clean(doc.page_content)
        if not text: continue
        page = int(doc.metadata.get("page", 0)) + 1
        pages.append(Document(page_content=text, metadata={"source": pdf_path.name, "page": page}))
    return pages

def split_documents(pages: list[Document], chunk_size: int = 1100, chunk_overlap: int = 150) -> list[Document]:
    """Split pages at paragraph/line/sentence boundaries; each chunk keeps its page and gets a stable chunk_id."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = chunk_id_for(chunk.metadata["page"], chunk.page_content)
        chunk.id = chunk.metadata["chunk_id"]
    return chunks
