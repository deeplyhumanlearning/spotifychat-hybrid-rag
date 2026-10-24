"""Build the LangChain FAISS store and populate Neo4j from the supplied architecture PDF.

Usage (inside the backend image):  python scripts/ingest.py [--skip-vectors] [--skip-graph] [--replace-graph]

The graph step only replaces existing :ArchitectureEntity data when --replace-graph is given; otherwise it aborts
before any embedding cost is incurred if that data is already present.
"""
import argparse, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the `app` package importable when run as a script

from langchain_core.documents import Document
from app.config import settings
from app.documents import load_pdf, split_documents
from app.vectorstore import build_embeddings, build_vector_store, save_vector_store, UnconfiguredEmbeddings

ROOT = Path("/app") if Path("/app/data/source").exists() else Path(__file__).resolve().parents[2]
PDF = Path(os.getenv("SOURCE_PDF", str(ROOT / "data/source/spotify_web_app_architecture.pdf")))
if "VECTOR_STORE_DIR" not in os.environ: settings.vector_store_dir = str(ROOT / "faiss_index")  # also works outside Docker

# Conservative extraction: explicit architecture terms only. Schema: Service, Database, API, Protocol, Microservice,
# Infrastructure, Framework, Storage, Component.
ENTITY_PATTERN = re.compile(r"(?i)\b(Next\.js|React|TypeScript|Tailwind CSS|FastAPI|Streamlit|NestJS|Node\.js|PostgreSQL|Redis|Elasticsearch|OpenSearch|ClickHouse|S3-compatible object storage|CDN|Docker|Kubernetes|Vault|GitHub Actions|Prometheus|Grafana|OpenTelemetry|API Gateway|Auth & Identity Service|User Profile Service|Catalog Service|Search & Discovery Index Service|Library & Playlist Service|Playback & Session Service|Streaming / Media Delivery Service|Recommendation & Personalization Service|Social & Activity Service|Subscription & Billing Service|Ads Service|Content Ingestion & Creator Portal Service|Media Processor Service|Analytics & Events Service|Notification Service|Admin & Backoffice Service|Kafka|NATS|RabbitMQ|HLS\.js|HTML5 Audio|OAuth2 / OIDC|JWT|WebSocket)\b")
ENTITY_TYPES = {"PostgreSQL":"Database","Redis":"Database","Elasticsearch":"Database","OpenSearch":"Database","ClickHouse":"Database","S3-compatible object storage":"Storage","CDN":"Infrastructure","Docker":"Infrastructure","Kubernetes":"Infrastructure","Vault":"Infrastructure","Prometheus":"Infrastructure","Grafana":"Infrastructure","OpenTelemetry":"Protocol","Kafka":"Protocol","NATS":"Protocol","RabbitMQ":"Protocol","OAuth2 / OIDC":"Protocol","JWT":"Protocol","WebSocket":"Protocol","API Gateway":"API","HLS.js":"Component","HTML5 Audio":"Component","Next.js":"Framework","React":"Framework","TypeScript":"Framework","Tailwind CSS":"Framework","NestJS":"Framework","Node.js":"Framework"}

def extract_entities(chunks: list[Document]):
    """Entities per chunk plus MENTIONED_WITH co-occurrence facts (explicitly not causal), each tied to its chunk_id."""
    entities = {}; facts = []
    for c in chunks:
        found = []
        for match in ENTITY_PATTERN.finditer(c.page_content):
            name = match.group(0); typ = ENTITY_TYPES.get(name, "Microservice" if "Service" in name else "Component")
            entities[name] = {"name": name, "type": typ, "source_page": c.metadata["page"]}; found.append(name)
        unique = list(dict.fromkeys(found))
        for i, a in enumerate(unique):
            for b in unique[i+1:]:
                facts.append({"subject": a, "object": b, "source_text": c.page_content[:800], "source_page": c.metadata["page"], "chunk_id": c.metadata["chunk_id"]})
    return list(entities.values()), facts

def connect_graph():
    if not settings.neo4j_uri or not settings.neo4j_password:
        raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD are required for graph ingestion.")
    from langchain_neo4j import Neo4jGraph
    from app.retrievers import DRIVER_CONFIG
    return Neo4jGraph(url=settings.neo4j_uri, username=settings.neo4j_username, password=settings.neo4j_password,
                      database=settings.neo4j_database, refresh_schema=False, driver_config=DRIVER_CONFIG)

def existing_entity_count(graph) -> int:
    return graph.query("MATCH (n:ArchitectureEntity) RETURN count(n) AS n")[0]["n"]

def populate_graph(graph, entities, facts):
    graph.query("CREATE CONSTRAINT architecture_entity_name IF NOT EXISTS FOR (n:ArchitectureEntity) REQUIRE n.name IS UNIQUE")
    # Scoped to this application's label only; the caller has already confirmed replacement is intended.
    graph.query("MATCH (n:ArchitectureEntity) DETACH DELETE n")
    for e in entities:
        label = e["type"] if e["type"].isidentifier() else "Component"
        graph.query(f"MERGE (n:ArchitectureEntity:{label} {{name:$name}}) SET n.source_page=$page, n.entity_type=$type",
                    params={"name": e["name"], "page": e["source_page"], "type": e["type"]})
    graph.query("""UNWIND $facts AS f
                   MATCH (a:ArchitectureEntity {name:f.subject}), (b:ArchitectureEntity {name:f.object})
                   MERGE (a)-[r:MENTIONED_WITH {chunk_id:f.chunk_id}]->(b)
                   SET r.label='MENTIONED_WITH', r.source_page=f.source_page, r.source_text=f.source_text""", params={"facts": facts})

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-vectors", action="store_true", help="do not rebuild the FAISS store")
    ap.add_argument("--skip-graph", action="store_true", help="do not touch Neo4j")
    ap.add_argument("--replace-graph", action="store_true", help="allow deleting existing :ArchitectureEntity nodes before reloading")
    args = ap.parse_args(argv)
    if not PDF.exists(): sys.exit(f"Source PDF not found: {PDF}")
    graph = None
    if not args.skip_graph:
        graph = connect_graph()
        count = existing_entity_count(graph)
        if count and not args.replace_graph:
            sys.exit(f"Neo4j already holds {count} :ArchitectureEntity nodes. Re-run with --replace-graph to delete and reload them, "
                     "or --skip-graph to rebuild only the FAISS store. Nothing was changed.")
    chunks = split_documents(load_pdf(PDF), settings.chunk_size, settings.chunk_overlap)
    if not chunks: sys.exit("No extractable text found in PDF.")
    if not args.skip_vectors:
        embeddings = build_embeddings(settings)
        if isinstance(embeddings, UnconfiguredEmbeddings): sys.exit("OPENAI_API_KEY is required for ingestion.")
        save_vector_store(build_vector_store(chunks, embeddings), settings)
        print(f"FAISS store written to {settings.vector_store_dir}/{settings.vector_index_name}.faiss|.pkl ({len(chunks)} chunks)")
    if graph is not None:
        entities, facts = extract_entities(chunks)
        populate_graph(graph, entities, facts)
        graph.close()
        print(f"Neo4j loaded: {len(entities)} entities, {len(facts)} conservative MENTIONED_WITH relationships.")
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
