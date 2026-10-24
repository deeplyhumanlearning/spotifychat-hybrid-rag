"""LangSmith tracing configuration.

LangChain traces every Runnable, retriever, prompt and chat-model call automatically once the standard
LangSmith environment variables are present, so no manual tracing code is needed in the pipeline itself.
This module only bridges values loaded from `.env` by pydantic-settings into the process environment
(Docker Compose already exports them) and disables tracing cleanly when it is misconfigured.
"""
import logging, os
from .config import Settings

log = logging.getLogger(__name__)

def _refresh_sdk_env_cache() -> None:
    # The LangSmith SDK memoises environment lookups; clear them so values applied here take effect immediately.
    from langsmith import utils
    for fn in ("get_env_var", "get_tracer_project"):
        cached = getattr(utils, fn, None)
        if hasattr(cached, "cache_clear"): cached.cache_clear()

def configure_langsmith(cfg: Settings) -> bool:
    """Apply LangSmith settings to the environment. Returns True when tracing is active. Never logs secrets."""
    enabled = False
    api_key = cfg.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY", "")
    if cfg.langsmith_tracing and not api_key:
        log.warning("LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is empty; tracing disabled.")
    elif cfg.langsmith_tracing:
        for var, value in (("LANGSMITH_API_KEY", api_key), ("LANGSMITH_PROJECT", cfg.langsmith_project), ("LANGSMITH_ENDPOINT", cfg.langsmith_endpoint)):
            if value and not os.environ.get(var): os.environ[var] = value
        enabled = True
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)  # legacy alias; LANGSMITH_TRACING is the single switch
    _refresh_sdk_env_cache()
    if enabled: log.info("LangSmith tracing enabled for project %r", os.environ.get("LANGSMITH_PROJECT", "default"))
    return enabled

def flush_traces() -> None:
    """Block until queued traces are sent (called on shutdown so the last requests are not lost)."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        log.debug("Trace flush skipped", exc_info=True)
