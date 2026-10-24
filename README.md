# SpotifyChat Hybrid RAG

A containerised assistant that answers questions about the included Spotify-like web application architecture PDF. Retrieval is local-first and hybrid (FAISS semantic search + Neo4j AuraDB graph facts, fused with Reciprocal Rank Fusion). The RAG pipeline is built with LangChain and can be traced end to end in LangSmith. Web search happens only after the assistant offers it and the user explicitly replies yes.

## Architecture

```text
Streamlit UI ──HTTP──► FastAPI /chat ── session + consent handling (app/main.py)
                                         │
                          LangChain LCEL pipeline (app/rag.py)  ── traced in LangSmith as one run per request
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
  FaissScoredRetriever                                         Neo4jGraphRetriever
  (langchain_community FAISS + langchain_openai embeddings)    (langchain_neo4j.Neo4jGraph, parameterised Cypher)
          └──────────────► rrf_fusion_and_validation (app/fusion.py) ◄──┘
                                         │ local context sufficient?
                     no ──► "Would you like me to perform a web search for this question?"  (no model call, no search)
                     yes ─► ChatPromptTemplate │ ChatOpenAI │ StrOutputParser ──► answer + sources + graph facts
  after "yes" ──► web_search (Serper/SerpAPI) ──► web ChatPromptTemplate │ ChatOpenAI │ StrOutputParser
```

### Where LangChain is used

| Responsibility | LangChain component | File |
|---|---|---|
| PDF loading | `langchain_community.document_loaders.PyPDFLoader` | `backend/app/documents.py` |
| Chunking | `langchain_text_splitters.RecursiveCharacterTextSplitter` → `Document` objects with `source`, `page`, `chunk_id`, `start_index` | `backend/app/documents.py` |
| Embeddings | `langchain_openai.OpenAIEmbeddings` | `backend/app/vectorstore.py` |
| Vector store | `langchain_community.vectorstores.FAISS` (`from_documents`, `save_local`, `load_local`, `similarity_search_with_score`) | `backend/app/vectorstore.py` |
| Vector retrieval | `FaissScoredRetriever(BaseRetriever)` | `backend/app/retrievers.py` |
| Graph access | `langchain_neo4j.Neo4jGraph` (app and ingestion) | `backend/app/retrievers.py`, `backend/scripts/ingest.py` |
| Graph retrieval | `Neo4jGraphRetriever(BaseRetriever)` | `backend/app/retrievers.py` |
| Orchestration | `RunnableParallel`, `RunnableLambda`, `RunnablePassthrough.assign`, LCEL `|` composition | `backend/app/rag.py` |
| Prompts | `ChatPromptTemplate` + `MessagesPlaceholder` for conversation history | `backend/app/prompts.py` |
| Generation | `langchain_openai.ChatOpenAI` + `StrOutputParser` | `backend/app/rag.py` |
| Tracing | LangChain's built-in LangSmith tracer (configured from env) | `backend/app/tracing.py` |

Plain Python is used where LangChain has no direct equivalent: RRF fusion, the context-sufficiency gate, session state, the consent state machine, and the Serper/SerpAPI HTTP client (wrapped in a `RunnableLambda` so it is traced). The OpenAI SDK is not called directly; only its exception types are imported to map timeouts/errors to HTTP responses. Graph Cypher is fixed and parameterised; no LLM-generated Cypher is executed.

### Hybrid retrieval and RRF

1. **FAISS** returns the top `RETRIEVAL_TOP_K` chunks by cosine similarity (vectors are L2-normalised and searched by inner product).
2. **Neo4j** returns relationship facts whose entity names, aliases or relationship label contain a meaningful query term (stopwords removed), ordered by number of matching terms. Each fact is linked back to the chunk it was extracted from: by `chunk_id` on graphs built by the current ingestion, or, for graphs built before that, by matching the `source_text` stored on the edge (whitespace-insensitive, so links survive re-chunking).
3. **Reciprocal Rank Fusion** treats both as rankings of chunks: `score(chunk) = Σ 1 / (RRF_K + rank)` over the FAISS list and the graph list (each chunk counted once per list, at its best rank). A chunk found by both retrievers outranks one found by either alone, and a chunk reached only through the graph can still enter the context. The top `RETRIEVAL_TOP_K` fused chunks plus the matching graph facts form the prompt context. Each source in the API response records `retrieval` (`FAISS`, `Neo4j` or `FAISS+Neo4j`), `rrf_score`, and the per-list ranks.
4. **Sufficiency gate**: at least one meaningful question term must appear in the fused context. If not, the API asks for web-search consent without calling the model. The model is also instructed to ask the same question when the passages do not actually answer; that also defers to consent.

If Neo4j is unreachable, retrieval continues vector-only (the connection is retried after 60 s). If the FAISS store is missing, there is no local context and the user is offered a web search.

### Web-search consent

`/chat` keeps a per-session `pending_query`. Only an affirmative reply (`yes`, `sure`, `go ahead`, …) to that session's pending offer triggers a search, and the search uses the **original** pending question, not the word "yes". `no` declines without searching; any other message cancels the offer and is answered as a new question. Consent is single-use and never shared across sessions.

## Project layout

```text
spotifychat-hybrid-rag/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI endpoints, sessions, consent flow
│   │   ├── rag.py           # LCEL pipeline (retrieval → fusion → prompt → model), error mapping
│   │   ├── retrievers.py    # FAISS and Neo4j LangChain retrievers
│   │   ├── fusion.py        # RRF + sufficiency validation
│   │   ├── documents.py     # PDF loader + text splitter
│   │   ├── vectorstore.py   # OpenAIEmbeddings + FAISS store load/save
│   │   ├── prompts.py       # ChatPromptTemplates
│   │   ├── websearch.py     # Serper / SerpAPI client
│   │   ├── tracing.py       # LangSmith configuration
│   │   ├── config.py, models.py
│   ├── scripts/ingest.py    # builds the FAISS store and loads Neo4j
│   ├── tests/               # pytest suite (no credentials needed)
│   ├── requirements.txt, requirements-dev.txt, pytest.ini, Dockerfile
├── frontend/app.py          # Streamlit UI
├── data/source/             # architecture PDF
├── faiss_index/             # generated vector store (mounted into the backend container)
├── nginx/                   # VPS reverse-proxy configuration
├── docker-compose.yml
└── .env.example
```

## Configuration

Copy `.env.example` to `.env` and fill in values. Never commit `.env`.

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | Embeddings and chat model |
| `OPENAI_MODEL`, `EMBEDDING_MODEL` | no | Defaults `gpt-4.1-mini`, `text-embedding-3-small` (must match the model used at ingestion) |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | for graph retrieval | AuraDB connection (`neo4j+s://…`) |
| `WEB_SEARCH_PROVIDER`, `SERPER_API_KEY` / `SERPAPI_API_KEY` | optional | External search after consent |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | optional | LangSmith tracing |
| `RETRIEVAL_TOP_K`, `RRF_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | no | Retrieval/fusion/chunking tuning |
| `SESSION_TTL_MINUTES`, `REQUEST_TIMEOUT_SECONDS`, `CORS_ORIGINS`, `STREAMLIT_PORT` | no | App settings |

## Build the knowledge base

Ingestion loads the PDF with `PyPDFLoader`, splits it with `RecursiveCharacterTextSplitter`, embeds the chunks with `OpenAIEmbeddings`, writes a LangChain FAISS store to `faiss_index/langchain.faiss` + `langchain.pkl`, and loads conservative `MENTIONED_WITH` co-occurrence relationships (with `chunk_id`) into Neo4j.

```powershell
docker compose build fastapi-backend
# Rebuild only the FAISS store; Neo4j is not touched:
docker compose run --rm --entrypoint python fastapi-backend scripts/ingest.py --skip-graph
```

Graph loading is guarded. If AuraDB already contains `:ArchitectureEntity` nodes, the script stops before doing anything unless you pass `--replace-graph`, which **deletes and reloads only `:ArchitectureEntity` nodes and their relationships**. Back up first and use a dedicated database:

```powershell
docker compose run --rm --entrypoint python fastapi-backend scripts/ingest.py --replace-graph
```

**Existing deployments:** an index built by the previous (pre-LangChain) ingestion (`faiss_index/index.faiss` + `chunks.json`) is still loaded, wrapped in LangChain's FAISS store without re-embedding; `/health` reports `"vector_store_format": "legacy"`. Run the `--skip-graph` command above to switch to the native store. Existing graph edges keep working with re-chunked text.

`faiss_index/langchain.pkl` is a pickle; the app loads it with `allow_dangerous_deserialization=True` because it is produced by this project's own ingestion. Never place index files from an untrusted source in `faiss_index/`.

## Run locally with Docker

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f fastapi-backend streamlit-frontend
```

Open `http://localhost:8501`. The API is not published to the host; check it with:

```powershell
docker compose exec fastapi-backend python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

`/health` reports `vector_index_loaded`, `vector_store_format`, `openai_configured`, `neo4j_configured` and `langsmith_tracing` (booleans and format names only, never secrets).

## LangSmith tracing

1. Create an API key in LangSmith and set in `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=…`, optionally `LANGSMITH_PROJECT` (default `spotifychat-hybrid-rag`) and `LANGSMITH_ENDPOINT` (only for EU/self-hosted).
2. Restart the backend: `docker compose up -d fastapi-backend`. `/health` should show `"langsmith_tracing": true`. If tracing is on but the key is empty, the backend logs a warning and runs with tracing off.
3. Ask a question in the UI and open the project in LangSmith. Each request appears as one run, `spotifychat_chat` (or `spotifychat_web_answer` after consent), tagged with the session id in metadata and containing: `hybrid_retrieval` → `faiss_vector_retriever` and `neo4j_graph_retriever` (retrieved documents), `rrf_fusion_and_validation` (fused chunks and validation), `grounded_answer` → `ChatPromptTemplate` (rendered prompt), `ChatOpenAI` (messages, tokens, latency) and `StrOutputParser`. Errors are recorded on the failing run.

No manual tracing code is needed; LangChain's tracer handles it. API keys are never part of run inputs/outputs, and traces are flushed on shutdown. To disable tracing set `LANGSMITH_TRACING=false`.

## Tests

The suite uses fakes for OpenAI (a recording chat model and deterministic embeddings), Neo4j and the search provider, so it needs no credentials or network:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

It covers PDF loading and chunking, embeddings configuration, FAISS save/load/retrieval and the legacy loader, Neo4j retrieval and failure handling, RRF, the sufficiency gate, prompt rendering and LCEL answer generation, the trace structure LangSmith receives, LangSmith configuration, consent (yes uses the original question, no never searches, single-use, per-session isolation), and error mapping when OpenAI, Neo4j, FAISS or the search provider is unavailable.

## Known limitations

- `langchain-community` (source of LangChain's FAISS vector store and `PyPDFLoader`) is marked by its maintainers as being sunset. 0.4.2 is the current release and works with `langchain-core` 1.7; there is no official standalone FAISS or PyPDF package yet. Revisit when one is published.
- Graph extraction is deliberately conservative: a fixed entity list and `MENTIONED_WITH` co-occurrence edges, which do not imply dependency.
- The sufficiency gate is lexical, not an entailment model; the model's own "not enough information" judgement is the second line of defence.
