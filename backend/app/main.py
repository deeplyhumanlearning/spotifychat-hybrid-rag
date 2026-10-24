import logging, time, re
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .models import ChatRequest, ChatResponse
from .prompts import CONSENT_QUESTION
from .rag import HybridRAGPipeline, PipelineError, build_pipeline
from .tracing import configure_langsmith, flush_traces

logging.basicConfig(level=logging.INFO)
log=logging.getLogger("spotifychat")
pipeline: HybridRAGPipeline | None = None
tracing_enabled=False
sessions: OrderedDict[str, dict] = OrderedDict()
MAX_SESSIONS=2000

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, tracing_enabled
    tracing_enabled=configure_langsmith(settings)
    pipeline=build_pipeline(settings)
    yield
    if pipeline: pipeline.close()
    flush_traces()

app=FastAPI(title="SpotifyChat Hybrid RAG API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()], allow_credentials=True, allow_methods=["GET","POST"], allow_headers=["*"])

@app.middleware("http")
async def request_limits(request: Request, call_next):
    if request.url.path == "/chat" and request.method == "POST":
        length=request.headers.get("content-length")
        if length and int(length)>20000: return JSONResponse(status_code=413,content={"detail":"Request body too large."})
    return await call_next(request)

@app.get("/health")
def health():
    return {"status":"ok","vector_index_loaded":bool(pipeline and pipeline.vector_loaded),"vector_store_format":pipeline.vector_format if pipeline else "missing",
            "openai_configured":bool(settings.openai_api_key),"neo4j_configured":bool(pipeline and pipeline.graph_configured),"langsmith_tracing":tracing_enabled}

@app.get("/ready")
def ready():
    if not pipeline or not pipeline.vector_loaded: raise HTTPException(503,"Vector index not ready. Run ingestion first.")
    return {"status":"ready"}

def get_session(sid):
    now=time.time(); s=sessions.get(sid)
    if s and now-s["updated"] > settings.session_ttl_minutes*60: sessions.pop(sid,None); s=None
    if not s: s={"history":[],"pending_query":None,"updated":now}; sessions[sid]=s
    s["updated"]=now; sessions.move_to_end(sid)
    while len(sessions)>MAX_SESSIONS: sessions.popitem(last=False)
    return s

def looks_like_yes(message: str) -> bool:
    return bool(re.fullmatch(r"\s*(yes|yes please|sure|go ahead|search the web|please search|do a web search|okay search|ok search)\s*[.!]*\s*",message,re.I))

def looks_like_no(message: str) -> bool:
    return bool(re.fullmatch(r"\s*(no|no thanks|not now|don't search|do not search|no web search)\s*[.!]*\s*",message,re.I))

def remember(session, *turns):
    session["history"].extend(turns); session["history"]=session["history"][-16:]; session["updated"]=time.time()

@app.post("/chat",response_model=ChatResponse)
async def chat(req: ChatRequest):
    query=req.message.strip()
    if not query or len(query)<2: raise HTTPException(422,"Please enter a question with at least two characters.")
    if not pipeline: raise HTTPException(503,"Retrieval engine is not initialized.")
    session=get_session(req.session_id)
    try:
        # Search only when the user explicitly says yes to this session's pending offer. A standalone yes is not enough.
        if session.get("pending_query"):
            if looks_like_yes(query):
                original=session["pending_query"]; session["pending_query"]=None
                result=await pipeline.aweb_answer(original,session["history"],req.session_id)
                remember(session,{"role":"user","content":original},{"role":"assistant","content":result.answer})
                return ChatResponse(answer=result.answer,status="web_answer" if result.results else "not_found",session_id=req.session_id,sources=result.results,web_used=bool(result.results),retrieval_validation={"local_context_sufficient":False,"web_search_consented":True})
            if looks_like_no(query):
                session["pending_query"]=None; answer="Understood. I will not perform a web search. Please ask another question about the local architecture document."
                session["history"].append({"role":"assistant","content":answer})
                return ChatResponse(answer=answer,status="not_found",session_id=req.session_id)
            # New question cancels the previous pending search offer and is handled as a fresh local query.
            session["pending_query"]=None
        outcome=await pipeline.achat(query,session["history"],req.session_id)
    except PipelineError as exc:
        raise HTTPException(exc.status_code,exc.detail) from exc
    if outcome.answer is None:
        answer="I could not find enough relevant information in the local knowledge base to answer this question. "+CONSENT_QUESTION
        session["pending_query"]=query; remember(session,{"role":"user","content":query},{"role":"assistant","content":answer})
        return ChatResponse(answer=answer,status="needs_web_consent",session_id=req.session_id,sources=outcome.chunks,graph_facts=outcome.graph_facts,retrieval_validation=outcome.validation)
    if outcome.needs_consent: session["pending_query"]=query
    remember(session,{"role":"user","content":query},{"role":"assistant","content":outcome.answer})
    return ChatResponse(answer=outcome.answer,status="needs_web_consent" if outcome.needs_consent else "answered",session_id=req.session_id,sources=outcome.chunks,graph_facts=outcome.graph_facts,retrieval_validation=outcome.validation)

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id,None)
    return {"status":"cleared","session_id":session_id}

@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("Unhandled request error")
    return JSONResponse(status_code=500,content={"detail":"An unexpected server error occurred. Please try again."})
