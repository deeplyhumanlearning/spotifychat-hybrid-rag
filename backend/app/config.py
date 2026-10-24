from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    serper_api_key: str = ""
    serpapi_api_key: str = ""
    web_search_provider: str = "serper"
    # LangChain FAISS store (written by scripts/ingest.py as <dir>/<name>.faiss + <name>.pkl).
    vector_store_dir: str = "/app/faiss_index"
    vector_index_name: str = "langchain"
    # Legacy raw-FAISS artifacts from the pre-LangChain ingestion; loaded only if the LangChain store is absent.
    vector_index_path: str = "/app/faiss_index/index.faiss"
    chunks_path: str = "/app/faiss_index/chunks.json"
    chunk_size: int = 1100
    chunk_overlap: int = 150
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.25
    rrf_k: int = 60
    session_ttl_minutes: int = 240
    request_timeout_seconds: float = 30.0
    cors_origins: str = "http://localhost:8501,https://spotifychat.labspace.tech"
    # LangSmith tracing (read by the LangChain/LangSmith SDK from these exact environment variable names).
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "spotifychat-hybrid-rag"
    langsmith_endpoint: str = ""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
