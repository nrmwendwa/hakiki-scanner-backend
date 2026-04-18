"""DuckDuckGo web search wrapper with trusted-domain tagging."""

from __future__ import annotations

import logging
from typing import Iterable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_TRUSTED_DOMAINS: tuple[str, ...] = (
    "gov.tz",
    "ac.tz",
    "go.tz",
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "apnews.com",
    "aljazeera.com",
    "theguardian.com",
    "nytimes.com",
    "thecitizen.co.tz",
    "mwananchi.co.tz",
    "dailynews.co.tz",
    "tcra.go.tz",
    "statehouse.go.tz",
    "who.int",
    "un.org",
    "worldbank.org",
    "wikipedia.org",
    "britannica.com",
)


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _is_trusted(url: str, trusted: Iterable[str]) -> bool:
    host = _domain(url)
    if not host:
        return False
    return any(host == d or host.endswith(f".{d}") for d in trusted)


def search_web(
    query: str,
    max_results: int = 6,
    trusted_domains: Iterable[str] | None = None,
) -> list[dict]:
    """Run a DDG text search and return a list of {title, url, snippet, trusted} dicts."""
    try:
        from ddgs import DDGS
    except Exception as exc:
        logger.warning("ddgs unavailable: %s", exc)
        return []

    trusted = tuple(trusted_domains) if trusted_domains else _DEFAULT_TRUSTED_DOMAINS

    results: list[dict] = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results) or []:
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                results.append(
                    {
                        "title": r.get("title", "") or "",
                        "url": url,
                        "snippet": r.get("body") or r.get("snippet") or "",
                        "trusted": _is_trusted(url, trusted),
                    }
                )
    except Exception as exc:
        logger.warning("DDG search failed for %r: %s", query, exc)
        return []

    results.sort(key=lambda r: (not r["trusted"],))
    return results
