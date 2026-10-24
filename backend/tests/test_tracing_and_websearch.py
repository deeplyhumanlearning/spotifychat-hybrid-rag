"""LangSmith configuration and the external search provider client."""
import asyncio, logging, os
import httpx, pytest
from app.config import Settings
from app.tracing import configure_langsmith
from app.websearch import search_configured, web_search

SECRET = "lsv2_pt_test_secret_value"

@pytest.fixture
def clean_env(monkeypatch):
    for var in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
        monkeypatch.setenv(var, "x"); monkeypatch.delenv(var)  # register for restore, then clear
    return monkeypatch

def test_tracing_enabled_exports_standard_variables(clean_env, caplog):
    caplog.set_level(logging.INFO)
    cfg = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key=SECRET, langsmith_project="proj", langsmith_endpoint="https://eu.api.smith.langchain.com")
    assert configure_langsmith(cfg) is True
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_API_KEY"] == SECRET
    assert os.environ["LANGSMITH_PROJECT"] == "proj" and os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"
    from langsmith.utils import tracing_is_enabled
    assert tracing_is_enabled()
    assert SECRET not in caplog.text

def test_tracing_without_key_is_disabled(clean_env, caplog):
    assert configure_langsmith(Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="")) is False
    assert os.environ["LANGSMITH_TRACING"] == "false" and "LANGSMITH_API_KEY is empty" in caplog.text

def test_tracing_off_by_default(clean_env):
    assert configure_langsmith(Settings(_env_file=None)) is False and os.environ["LANGSMITH_TRACING"] == "false"
    from langsmith.utils import tracing_is_enabled
    assert not tracing_is_enabled()

def test_existing_environment_key_is_respected(clean_env):
    clean_env.setenv("LANGSMITH_API_KEY", "from-env")
    assert configure_langsmith(Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key="")) is True
    assert os.environ["LANGSMITH_API_KEY"] == "from-env"

def _transport(seen, payload, status=200):
    def handler(request):
        seen.append(request); return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)

def test_serper_results_are_normalised():
    seen = []; cfg = Settings(_env_file=None, serper_api_key="serper-key")
    payload = {"organic": [{"title": f"T{i}", "snippet": "S", "link": f"https://e.com/{i}"} for i in range(7)]}
    results = asyncio.run(web_search("india pm", cfg, _transport(seen, payload)))
    assert len(results) == 5 and results[0] == {"title": "T0", "snippet": "S", "url": "https://e.com/0"}
    assert seen[0].headers["X-API-KEY"] == "serper-key" and b"india pm" in seen[0].content

def test_serpapi_provider():
    seen = []; cfg = Settings(_env_file=None, web_search_provider="serpapi", serpapi_api_key="k")
    results = asyncio.run(web_search("q", cfg, _transport(seen, {"organic_results": [{"title": "A", "snippet": "B", "link": "https://a"}]})))
    assert results == [{"title": "A", "snippet": "B", "url": "https://a"}] and seen[0].url.host == "serpapi.com"

def test_no_provider_key_means_no_request():
    seen = []; cfg = Settings(_env_file=None)
    assert not search_configured(cfg)
    assert asyncio.run(web_search("q", cfg, _transport(seen, {}))) == [] and seen == []

def test_provider_http_error_raises():
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(web_search("q", Settings(_env_file=None, serper_api_key="k"), _transport([], {}, status=500)))
