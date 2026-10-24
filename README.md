# SpotifyChat Hybrid RAG

A containerised assistant that answers questions about the included Spotify-like web application architecture PDF. Retrieval is local-first and hybrid (FAISS semantic search + Neo4j AuraDB graph facts, fused with Reciprocal Rank Fusion). The RAG pipeline is built with LangChain and can be traced end to end in LangSmith. Web search happens only after the assistant offers it and the user explicitly replies yes.

## Demo

[Demo Video](https://github.com/user-attachments/assets/0bea95ae-596e-45b7-9c53-5ba5e87c8afa)

## Tech Stack

- **Python** – Application development
- **Streamlit** – User interface
- **FastAPI** – Backend API
- **LangChain** – RAG pipeline and LLM integration
- **FAISS** – Vector similarity search
- **Neo4j AuraDB** – Graph-based retrieval
- **OpenAI** – Embeddings and language model
- **Docker Compose** – Containerized deployment

## How It Works

1. The user enters a question in the Streamlit interface.
2. The backend retrieves relevant information from the FAISS vector store and Neo4j graph.
3. The retrieved results are combined using Reciprocal Rank Fusion (RRF).
4. LangChain passes the relevant context to the language model to generate an answer.
5. Web search can be used when enabled and when the user gives permission.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key
- Neo4j AuraDB credentials, if using graph retrieval
- A web search API key, if using web search

### Configuration

1. Clone the repository.
2. Copy `.env.example` to `.env`.
3. Add the required API keys and configuration values to `.env`.

**Never commit your `.env` file or expose API keys in your repository.**

### Run the Application

Build and start the application:

```bash
docker compose up -d --build
```

Check the running containers:

```bash
docker compose ps
```

The Streamlit interface is available at `http://localhost:8501` when running locally with the default configuration.

## Knowledge Base

Place the source documents in the appropriate data directory and run the project's ingestion process to prepare the data for retrieval.

The application uses FAISS for vector search and Neo4j for graph-based retrieval.

## Optional: LangSmith Tracing

LangSmith can be enabled to inspect and debug LangChain runs.

Tracing is optional. The application can run without LangSmith, and a LangSmith API key is not required unless you want to enable tracing.

To enable it, configure the relevant LangSmith environment variables in your `.env` file. Keep the API key private and do not commit it to GitHub.

## Notes

- The application requires valid API credentials for the services you enable.
- Web search depends on the configured search provider and user permission.
- LangSmith tracing is optional and is not required for the core RAG pipeline.
- Deployment details and private service URLs are intentionally not included in this README.
