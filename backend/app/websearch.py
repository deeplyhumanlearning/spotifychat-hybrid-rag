"""External web search (Serper or SerpAPI). Only ever called after the user consents."""
import httpx
from .config import Settings

def search_configured(cfg: Settings) -> bool:
    return bool((cfg.web_search_provider.lower() == "serpapi" and cfg.serpapi_api_key) or cfg.serper_api_key)

async def web_search(query: str, cfg: Settings, transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    """Top-5 organic results as {title, snippet, url}. Raises on provider/network errors; [] when no key is configured."""
    provider = cfg.web_search_provider.lower()
    async with httpx.AsyncClient(timeout=12.0, transport=transport) as http:
        if provider == "serpapi" and cfg.serpapi_api_key:
            r = await http.get("https://serpapi.com/search.json", params={"engine": "google", "q": query, "api_key": cfg.serpapi_api_key, "num": 5})
            r.raise_for_status(); data = r.json()
            return [{"title": x.get("title", ""), "snippet": x.get("snippet", ""), "url": x.get("link", "")} for x in data.get("organic_results", [])[:5]]
        if cfg.serper_api_key:
            r = await http.post("https://google.serper.dev/search", headers={"X-API-KEY": cfg.serper_api_key, "Content-Type": "application/json"}, json={"q": query, "num": 5})
            r.raise_for_status(); data = r.json()
            return [{"title": x.get("title", ""), "snippet": x.get("snippet", ""), "url": x.get("link", "")} for x in data.get("organic", [])[:5]]
    return []
