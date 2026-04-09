from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient


OFFICIAL_DOMAINS = {"pib.gov.in", "india.gov.in"}
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which", "about",
    "income", "below", "above", "category", "state", "scheme", "schemes", "apply",
}


def _is_official_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if host in OFFICIAL_DOMAINS:
        return True
    return host.endswith(".gov.in")


class WebSearchAgent:
    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
        key = api_key or os.getenv("TAVILY_API_KEY")
        self.available = bool(key)
        self.client = TavilyClient(api_key=key) if key else None
        self.last_status: Dict[str, str] = {
            "status": "ready" if key else "missing_api_key",
            "detail": "Tavily client initialized." if key else "Set TAVILY_API_KEY in .env.",
        }

    def search_official(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        if not self.available or self.client is None:
            self.last_status = {
                "status": "missing_api_key",
                "detail": "Official web search is disabled because TAVILY_API_KEY is not configured.",
            }
            return []

        try:
            response = self.client.search(
                query=query,
                max_results=8,
                search_depth="basic",
                include_raw_content=False,
            )
            raw_results = response.get("results", [])
        except Exception as exc:
            self.last_status = {
                "status": "search_error",
                "detail": f"Official web search failed: {exc}",
            }
            return []

        query_terms = {
            token for token in re.findall(r"[a-zA-Z]{4,}", query.lower()) if token not in STOPWORDS
        }

        ranked: List[Dict[str, str | int]] = []
        for item in raw_results:
            url = item.get("url", "")
            if not _is_official_url(url):
                continue
            title = item.get("title", "Untitled")
            snippet = item.get("content", "")[:280]
            haystack = f"{title} {snippet} {url}".lower()
            relevance = sum(1 for t in query_terms if t in haystack)
            ranked.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": (urlparse(url).hostname or "").lower(),
                    "relevance": relevance,
                }
            )

        ranked.sort(key=lambda x: int(x.get("relevance", 0)), reverse=True)
        filtered = [
            {
                "title": str(item.get("title", "Untitled")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", "")),
                "domain": str(item.get("domain", "")),
            }
            for item in ranked[:max_results]
        ]
        if filtered:
            self.last_status = {
                "status": "ok",
                "detail": f"Found {len(filtered)} official web result(s).",
            }
        elif raw_results:
            self.last_status = {
                "status": "filtered_out",
                "detail": "Search returned results, but none matched the official-domain filter well enough.",
            }
        else:
            self.last_status = {
                "status": "no_results",
                "detail": "Search completed but returned no results.",
            }
        return filtered
