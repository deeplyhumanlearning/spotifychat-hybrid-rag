import os, uuid, html
import httpx
import streamlit as st

API_URL=os.getenv("BACKEND_URL","http://fastapi-backend:8000")
st.set_page_config(page_title="SpotifyChat · Architecture Assistant", page_icon="🎧", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme (colors forced explicitly so it renders the same under Streamlit's light or dark base) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  color-scheme:dark;
  --bg:#0b0f14;--surface:#11161d;--surface-2:#161c24;--surface-3:#1c232d;
  --border:#232b36;--border-strong:#2f3946;
  --text:#e6edf3;--text-2:#a3b0bd;--text-3:#6e7b88;
  --accent:#1db954;--accent-hover:#1ed760;--accent-soft:rgba(29,185,84,.12);
  --warn:#f0b429;--danger:#f25f5c;
  --radius:12px;--radius-sm:8px;
}
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;color:var(--text);font-family:'Inter',system-ui,-apple-system,sans-serif}
.stApp p,.stApp li,.stApp span,.stApp label,.stApp div[data-testid="stMarkdownContainer"]{color:var(--text)}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--text);font-family:'Inter',sans-serif;letter-spacing:-.02em}
.stApp code{font-family:'JetBrains Mono',monospace;background:var(--surface-3);color:#9be9b5;border-radius:6px;padding:.1rem .35rem;font-size:.82em}
.stApp a{color:var(--accent-hover)}
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;height:0}
[data-testid="stElementContainer"]:has(iframe[height="0"]),.element-container:has(iframe[height="0"]){display:none}
header[data-testid="stHeader"]{background:var(--bg)!important}
.block-container{padding-top:1.2rem;padding-bottom:9rem;max-width:820px}

/* Sidebar */
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:700;font-size:1.05rem;color:var(--text)}
.brand-mark{width:30px;height:30px;border-radius:8px;background:var(--accent);display:grid;place-items:center;color:#04120a!important;font-size:.95rem;font-weight:800}
.brand-sub{color:var(--text-3)!important;font-size:.78rem;margin:.35rem 0 1.1rem}
.side-label{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--text-3)!important;margin:1.4rem 0 .55rem}
.status-row{display:flex;justify-content:space-between;align-items:center;padding:.45rem 0;border-bottom:1px solid var(--border);font-size:.82rem}
.status-row:last-child{border-bottom:none}
.status-row .k{color:var(--text-2)!important}
.dot{display:inline-flex;align-items:center;gap:.4rem;font-size:.76rem;font-weight:500}
.dot::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.dot.ok{color:var(--accent)!important}.dot.bad{color:var(--danger)!important}.dot.warn{color:var(--warn)!important}
.kb{background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.75rem .85rem}
.kb-title{font-weight:600;font-size:.86rem}
.kb-meta{color:var(--text-3)!important;font-size:.75rem;margin-top:.2rem}
.tag{display:inline-block;font-size:.68rem;font-weight:500;color:var(--text-2)!important;background:var(--surface-3);border:1px solid var(--border);border-radius:999px;padding:.12rem .5rem;margin:.45rem .25rem 0 0}
.side-foot{color:var(--text-3)!important;font-size:.72rem;line-height:1.5;margin-top:1.4rem}
.side-foot code{font-size:.7rem}

/* Buttons */
.stButton>button{background:var(--surface-2);color:var(--text);border:1px solid var(--border-strong);border-radius:var(--radius-sm);font-weight:500;font-size:.85rem;padding:.5rem .9rem;transition:all .15s ease}
.stButton>button:hover{border-color:var(--accent);color:var(--accent-hover);background:var(--surface-3)}
.stButton>button:focus:not(:active){border-color:var(--accent);color:var(--text);box-shadow:none}
.stButton>button p{color:inherit!important}
section[data-testid="stSidebar"] .stButton>button{background:var(--accent);color:#04120a;border-color:var(--accent);font-weight:600}
section[data-testid="stSidebar"] .stButton>button:hover{background:var(--accent-hover);color:#04120a}

/* Top bar */
.topbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding-bottom:.9rem;margin-bottom:.4rem;border-bottom:1px solid var(--border)}
.topbar-title{font-size:1rem;font-weight:600;color:var(--text)}
.topbar-title span{color:var(--text-3)!important;font-weight:400;margin-left:.4rem;font-size:.88rem}
.badge{white-space:nowrap;font-size:.74rem;font-weight:500;border-radius:999px;padding:.28rem .7rem;border:1px solid var(--border-strong);background:var(--surface)}

/* Empty state (centered welcome, ChatGPT-style) */
.empty{text-align:center;padding:11vh 1rem 1.6rem}
.empty-icon{width:56px;height:56px;margin:0 auto 1.1rem;border-radius:16px;background:var(--accent-soft);color:var(--accent)!important;display:grid;place-items:center;font-size:1.6rem}
.empty h2{font-size:2rem;font-weight:600;margin:0;padding:0;letter-spacing:-.03em}
.empty p{color:var(--text-2)!important;font-size:1rem;margin:.6rem auto 0;max-width:520px;line-height:1.6}
[data-testid="stHorizontalBlock"] .stButton>button{min-height:64px;justify-content:flex-start;text-align:left;padding:.8rem 1rem;border-radius:var(--radius);background:var(--surface);font-size:.9rem;color:var(--text-2)}
[data-testid="stHorizontalBlock"] .stButton>button:hover{color:var(--text);background:var(--surface-2)}

/* Chat */
[data-testid="stChatMessage"]{display:flex;align-items:flex-start;background:transparent!important;padding:1.1rem 0;margin:0;border:none;border-radius:0;gap:1rem}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,[data-testid="stChatMessage"] li{font-size:1rem;line-height:1.75}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3{font-size:1.1rem;margin-top:.8rem}
/* User turn: right-aligned bubble, no avatar */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse;justify-content:flex-start}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"]{display:none}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{flex:0 1 auto;width:auto;max-width:78%;background:var(--surface-3);border-radius:20px;padding:.7rem 1.15rem;margin-left:auto!important;margin-right:0!important}
[data-testid="stChatMessageAvatarAssistant"]{background:var(--accent)!important;color:#04120a!important;border-radius:50%;width:2rem;height:2rem}
[data-testid="stChatMessageAvatarAssistant"] svg{color:inherit;fill:currentColor}
/* Composer */
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:var(--bg)!important}
[data-testid="stBottomBlockContainer"]{max-width:860px;margin:0 auto;padding:1rem 1rem 2rem!important}
[data-testid="stChatInput"]{background:var(--surface-2)!important;border:1px solid var(--border-strong)!important;border-radius:28px!important;box-shadow:0 6px 28px rgba(0,0,0,.45)!important;padding:.65rem .7rem .65rem 1.4rem!important;min-height:76px;display:flex;align-items:center}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px var(--accent-soft),0 6px 28px rgba(0,0,0,.45)!important}
/* Streamlit nests several wrappers (baseweb textarea / base-input) that carry their own light background */
[data-testid="stChatInput"] div,[data-testid="stChatInput"] [data-baseweb="textarea"],[data-testid="stChatInput"] [data-baseweb="base-input"]{background:transparent!important;background-color:transparent!important;border:none!important;box-shadow:none!important}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--text)!important;-webkit-text-fill-color:var(--text);caret-color:var(--accent);font-size:1.08rem!important;line-height:1.55!important;min-height:52px!important;padding:.85rem 0!important}
[data-testid="stChatInput"] textarea::placeholder{color:var(--text-3)!important;-webkit-text-fill-color:var(--text-3)}
[data-testid="stChatInputSubmitButton"]{background:var(--accent)!important;border:0!important;outline:none!important;box-shadow:none!important;border-radius:50%!important;width:44px!important;height:44px!important;min-width:44px;align-self:center;color:#04120a!important}
[data-testid="stChatInputSubmitButton"] svg{fill:#04120a!important;color:#04120a!important}
[data-testid="stChatInputSubmitButton"]:disabled{background:var(--surface-3)!important}
[data-testid="stChatInputSubmitButton"]:disabled svg{fill:var(--text-3)!important;color:var(--text-3)!important}
[data-testid="stSpinner"] *{color:var(--text-2)!important;font-size:.85rem}

/* Alerts */
[data-testid="stAlert"]{background:rgba(240,180,41,.08)!important;border:1px solid rgba(240,180,41,.3);border-radius:var(--radius-sm)}
[data-testid="stAlert"] p{color:#f5d27a!important;font-size:.86rem}

/* Retrieval context */
.section-title{display:flex;align-items:center;justify-content:space-between;margin:2.2rem 0 .8rem}
.section-title h3{font-size:.95rem;font-weight:600;margin:0;padding:0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:1rem}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.7rem .85rem}
.stat-v{font-size:1.05rem;font-weight:600}
.stat-l{color:var(--text-3)!important;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;margin-top:.15rem}
@media (max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}.empty h2{font-size:1.5rem}.empty{padding-top:5vh}}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;margin-top:1.4rem}
[data-testid="stExpander"] summary p{color:var(--text-2)!important;font-size:.88rem;font-weight:500}
[data-testid="stExpander"] summary:hover p{color:var(--text)!important}
[data-testid="stExpander"] summary svg{fill:var(--text-3)}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;background:transparent;border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--text-2);padding:.5rem .8rem}
.stTabs [data-baseweb="tab"] p{color:inherit!important;font-size:.85rem}
.stTabs [aria-selected="true"]{color:var(--text)!important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent)}
.stTabs [data-baseweb="tab-border"]{display:none}
.src{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:.85rem 1rem;margin:.6rem 0}
.src-head{display:flex;justify-content:space-between;align-items:center;gap:.8rem;margin-bottom:.45rem}
.src-name{font-weight:600;font-size:.84rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.src-meta{color:var(--text-3)!important;font-size:.74rem;white-space:nowrap;font-family:'JetBrains Mono',monospace}
.src-body{color:var(--text-2)!important;font-size:.84rem;line-height:1.6;max-height:9.6em;overflow:auto}
.src a{font-size:.78rem;text-decoration:none}
.triple{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem;font-size:.84rem}
.node{background:var(--surface-3);border:1px solid var(--border-strong);border-radius:6px;padding:.15rem .5rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--text)!important}
.rel{color:var(--accent)!important;font-family:'JetBrains Mono',monospace;font-size:.74rem;font-weight:500}
.empty-note{color:var(--text-3)!important;font-size:.84rem;padding:.8rem 0}
.foot{text-align:center;color:var(--text-3)!important;font-size:.74rem;margin-top:2rem}
</style>
""",unsafe_allow_html=True)

if "session_id" not in st.session_state: st.session_state.session_id=str(uuid.uuid4())
if "messages" not in st.session_state: st.session_state.messages=[]
if "last_sources" not in st.session_state: st.session_state.last_sources=[]
if "last_graph" not in st.session_state: st.session_state.last_graph=[]
if "last_validation" not in st.session_state: st.session_state.last_validation={}

try:
    health=httpx.get(f"{API_URL}/health",timeout=2).json()
except Exception: health={"status":"unavailable","vector_index_loaded":False,"openai_configured":False}

def esc(v): return html.escape(str(v))
def dot(ok,yes="Online",no="Offline",warn=False): return f'<span class="dot {"ok" if ok else ("warn" if warn else "bad")}">{yes if ok else no}</span>'

backend_ok=health.get("status") not in (None,"unavailable")
index_ok=bool(health.get("vector_index_loaded"))
openai_ok=bool(health.get("openai_configured"))

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">♪</div>SpotifyChat</div><div class="brand-sub">Architecture assistant · Hybrid RAG</div>',unsafe_allow_html=True)
    if st.button("＋  New conversation",use_container_width=True):
        try: httpx.delete(f"{API_URL}/sessions/{st.session_state.session_id}",timeout=3)
        except Exception: pass
        st.session_state.session_id=str(uuid.uuid4()); st.session_state.messages=[]; st.session_state.last_sources=[]; st.session_state.last_graph=[]; st.session_state.last_validation={}; st.rerun()

    st.markdown('<div class="side-label">System status</div>'
        f'<div class="status-row"><span class="k">Backend API</span>{dot(backend_ok)}</div>'
        f'<div class="status-row"><span class="k">Vector index</span>{dot(index_ok,"Loaded","Not loaded",warn=backend_ok)}</div>'
        f'<div class="status-row"><span class="k">LLM provider</span>{dot(openai_ok,"Configured","Missing key",warn=backend_ok)}</div>',unsafe_allow_html=True)

    st.markdown('<div class="side-label">Knowledge base</div>'
        '<div class="kb"><div class="kb-title">Spotify-like Web App</div><div class="kb-meta">Architecture document</div>'
        '<span class="tag">FAISS</span><span class="tag">Neo4j</span><span class="tag">RRF fusion</span></div>',unsafe_allow_html=True)

    st.markdown(f'<div class="side-foot">Answers come from the local knowledge base first. Web search is used only with your approval.<br><br>Session <code>{esc(st.session_state.session_id[:8])}</code></div>',unsafe_allow_html=True)

# ---------- Header ----------
if backend_ok and index_ok: badge='<span class="badge"><span class="dot ok">Ready</span></span>'
elif backend_ok: badge='<span class="badge"><span class="dot warn">Index not loaded</span></span>'
else: badge='<span class="badge"><span class="dot bad">Backend offline</span></span>'
st.markdown(f'<div class="topbar"><div class="topbar-title">Architecture Assistant<span>Spotify-like Web App</span></div>{badge}</div>',unsafe_allow_html=True)

if not index_ok:
    st.warning("The local vector index is not loaded yet. Configure OPENAI_API_KEY and run the ingestion script before asking questions.")

# ---------- Conversation ----------
SUGGESTIONS=["What are the core services and how do they communicate?","Which databases are used, and for what data?","How does user authentication work?","How is audio media processed and delivered?"]

if not st.session_state.messages:
    st.markdown('<div class="empty"><div class="empty-icon">♪</div><h2>What can I help you with?</h2><p>Ask anything about the architecture: services, databases, data flows, authentication, or media processing.</p></div>',unsafe_allow_html=True)
    cols=st.columns(2)
    for i,s in enumerate(SUGGESTIONS):
        if cols[i%2].button(s,key=f"sugg_{i}",use_container_width=True):
            st.session_state.pending_prompt=s; st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

prompt=st.chat_input("Ask about services, data stores, protocols, or features…")
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt=st.session_state.pending_prompt; st.session_state.pending_prompt=None

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge graph and vector index…"):
            try:
                response=httpx.post(f"{API_URL}/chat",json={"session_id":st.session_state.session_id,"message":prompt},timeout=45)
                if response.status_code>=400: answer=response.json().get("detail","The request could not be completed."); data={}
                else: data=response.json(); answer=data.get("answer","No answer returned.")
            except httpx.TimeoutException: answer="The request timed out. Please retry."; data={}
            except Exception: answer="The backend is currently unavailable. Check that the Docker services are running."; data={}
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.last_sources=data.get("sources",[]); st.session_state.last_graph=data.get("graph_facts",[]); st.session_state.last_validation=data.get("retrieval_validation",{})
    st.rerun()

# No custom auto-scroll: leave the viewport position unchanged after responses.

# ---------- Retrieval context ----------
sources,graph,val=st.session_state.last_sources,st.session_state.last_graph,st.session_state.last_validation
if sources or graph or val:
    with st.expander(f"Retrieval context for last answer · {len(sources)} sources · {len(graph)} graph facts",expanded=False):
        if val:
            sufficient=val.get("local_context_sufficient")
            st.markdown('<div class="stats">'
                f'<div class="stat"><div class="stat-v">{dot(sufficient,"Sufficient","Limited",warn=True)}</div><div class="stat-l">Local context</div></div>'
                f'<div class="stat"><div class="stat-v">{esc(val.get("vector_chunks",0))}</div><div class="stat-l">Vector chunks</div></div>'
                f'<div class="stat"><div class="stat-v">{esc(val.get("graph_facts",0))}</div><div class="stat-l">Graph facts</div></div>'
                f'<div class="stat"><div class="stat-v">{esc(val.get("fusion","RRF"))}</div><div class="stat-l">Fusion</div></div>'
                '</div>',unsafe_allow_html=True)

        tab_src,tab_graph=st.tabs([f"Source chunks ({len(sources)})",f"Graph facts ({len(graph)})"])
        with tab_src:
            if not sources: st.markdown('<div class="empty-note">No source chunks were retrieved.</div>',unsafe_allow_html=True)
            for i,src in enumerate(sources,1):
                score=src.get("score",src.get("rrf_score")) or 0
                link=f'<div style="margin-top:.5rem"><a href="{esc(src["url"])}" target="_blank">Open web source ↗</a></div>' if src.get("url") else ""
                st.markdown(f'<div class="src"><div class="src-head"><div class="src-name">{i}. {esc(src.get("source","Source"))}</div>'
                    f'<div class="src-meta">p.{esc(src.get("page","?"))} · {float(score):.3f}</div></div>'
                    f'<div class="src-body">{esc(src.get("text",src.get("snippet","No source text available.")))}</div>{link}</div>',unsafe_allow_html=True)
        with tab_graph:
            if not graph: st.markdown('<div class="empty-note">No matching graph relationships were retrieved.</div>',unsafe_allow_html=True)
            for g in graph:
                types=f'{", ".join(g.get("subject_types",[])) or "Entity"} → {", ".join(g.get("object_types",[])) or "Entity"} · p.{g.get("page","?")}'
                body=f'<div class="src-body" style="margin-top:.5rem">{esc(g["source_text"][:400])}</div>' if g.get("source_text") else ""
                st.markdown(f'<div class="src"><div class="src-head"><div class="triple"><span class="node">{esc(g.get("subject","?"))}</span>'
                    f'<span class="rel">— {esc(g.get("relation","RELATED_TO"))} →</span><span class="node">{esc(g.get("object","?"))}</span></div></div>'
                    f'<div class="src-meta">{esc(types)}</div>{body}</div>',unsafe_allow_html=True)

st.markdown('<div class="foot">Answers are grounded in the architecture document where possible. Verify critical details against the source.</div>',unsafe_allow_html=True)
