"""
APEX Web Search & Real-Time Fact-Grounding Module.
Searches the live web/internet to retrieve authoritative factual evidence,
ensuring strict factual correctness and preventing LLM hallucination.
"""

import os
import re
import html
import time
import urllib.parse
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import requests

from .logger import get_logger

logger = get_logger("web_search")

# In-memory LRU cache with TTL (1 hour)
_WEB_CACHE: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}
CACHE_TTL_SECONDS = 3600

# Explicit web triggers
EXPLICIT_WEB_PATTERNS = [
    r"\b(search\s+(the\s+)?(web|internet|online|google))\b",
    r"\b(look\s+up\s+(online|on\s+the\s+(web|internet)))\b",
    r"\b(browse\s+(the\s+)?(web|internet))\b",
    r"\b(current\s+(news|price|weather|events|status|score|president|prime\s+minister|ceo))\b",
    r"\b(latest\s+(news|update|developments|discoveries|release|version|winner|score))\b",
    r"\b(today\'?s?\s+(news|weather|events?|headlines?|stocks?|price|scores?|updates?))\b",
    r"\b(happened\s+(today|yesterday|this\s+week))\b",
    r"\b(happening\s+today)\b",
    r"\b(in\s+202[4-9])\b"
]
EXPLICIT_WEB_REGEX = re.compile("|".join(EXPLICIT_WEB_PATTERNS), re.IGNORECASE)

# Factual question patterns that require world knowledge / factual verification
FACTUAL_QUERY_PATTERNS = [
    r"\b(difference\s+between)\b",
    r"\b(explain\s+.*?(difference|concept|mechanism|process|how|why))\b",
    r"^(what\s+is\s+the\s+(capital|population|longest|largest|highest|deepest|hottest|coldest|height|distance|speed|currency|founder|ceo|president|winner|formula|origin|cause|difference)\b)",
    r"^(who\s+(is|was|are|were|won|invented|discovered|created|founded|directed|wrote|built|defeated)\b)",
    r"^(where\s+(is|are|was|were|located|situated)\b)",
    r"^(when\s+(was|were|did|is|will|happened)\b)",
    r"^(why\s+(did|does|do|is|are)\b)",
    r"^(how\s+(does|do|did|works?|many|much|far|old|tall|long)\b)",
    r"^(define|meaning\s+of|definition\s+of)\b"
]
FACTUAL_QUERY_REGEX = re.compile("|".join(FACTUAL_QUERY_PATTERNS), re.IGNORECASE)

# Casual pleasantries and creative requests that should NOT trigger web search
CHITCHAT_PATTERNS = [
    r"\b(hi|hello|hey|yo|howdy|sup|good\s+(morning|afternoon|evening|night))\b",
    r"\b(how\s+are\s+you|how\s+are\s+you\s+doing|how\'s\s+it\s+going|how\s+do\s+you\s+feel)\b",
    r"\b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)\b",
    r"\b(can\s+you\s+)?(tell\s+me\s+)?(a\s+)?(funny\s+)?(joke|riddle|pun|story)\b",
    r"\b(make\s+me\s+laugh|say\s+something\s+funny)\b",
    r"\b(write\s+(a\s+)?(poem|song|rap|haiku|limerick|short\s+story|script\s+to|python\s+script|code\s+to|function\s+to))\b",
    r"\b(thank|thanks|bye|goodbye|see\s+you|cheers)\b"
]
CHITCHAT_REGEX = re.compile("|".join(CHITCHAT_PATTERNS), re.IGNORECASE)

# Episodic memory indicators
MEMORY_INDICATORS = {
    "remember", "earlier", "before", "previously", "last time", "again",
    "yesterday", "we talked", "you said", "i told", "my name", "my dog",
    "my project", "my pet", "did we", "what was my", "do you know me"
}


def clean_text(text: str) -> str:
    """Normalize extracted web text and eliminate problematic unicode characters."""
    if not text:
        return ""
    text = html.unescape(text)
    # Replace non-breaking spaces, unicode hyphens, quotes with standard ASCII equivalents
    text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_search_terms(query: str) -> str:
    """Strip conversational filler to form a clean search query."""
    q = query.strip()
    # Remove phrases like 'search the web for', 'can you tell me', 'look up'
    q = re.sub(r'^(please\s+)?(can\s+you\s+)?(search\s+(the\s+)?(web|internet|online)\s+(for|about)?)\s*', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^(please\s+)?(can\s+you\s+)?(look\s+up\s+(online|on\s+the\s+web)?\s*(for|about)?)\s*', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^(tell\s+me\s+(about\s+)?|what\s+is\s+|who\s+is\s+|who\s+was\s+)', '', q, flags=re.IGNORECASE)
    cleaned = q.strip(' ?.!,"\'')
    return cleaned if len(cleaned) >= 3 else query.strip(' ?.!,"\'')


def search_duckduckgo(query: str, max_results: int = 4, timeout: int = 4) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo HTML endpoint and extract top organic results.
    """
    search_q = extract_search_terms(query)
    endpoints = [
        f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(search_q)}",
        f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(search_q)}"
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7"
    }
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for result in soup.find_all("div", class_="result"):
                snippet_elem = result.find("a", class_="result__snippet")
                heading_elem = result.find("h2", class_="result__title")
                title_elem = result.find("a", class_="result__url")

                title = heading_elem.get_text(strip=True) if heading_elem else ""
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                raw_url = title_elem.get("href", "") if title_elem else ""
                actual_url = raw_url
                if "uddg=" in raw_url:
                    try:
                        actual_url = urllib.parse.unquote(raw_url.split("uddg=")[-1].split("&")[0])
                    except Exception:
                        actual_url = raw_url

                if snippet:
                    domain = urllib.parse.urlparse(actual_url).netloc.replace("www.", "") if actual_url else "web"
                    results.append({
                        "title": clean_text(title),
                        "snippet": clean_text(snippet),
                        "url": actual_url,
                        "domain": domain
                    })
                if len(results) >= max_results:
                    break
            if results:
                return results
        except Exception as e:
            logger.debug(f"DuckDuckGo search error on {url}: {e}")
            continue
    return []


def search_wikipedia(query: str, max_results: int = 3, timeout: int = 3) -> List[Dict[str, str]]:
    """
    Authoritative encyclopedic search using Wikipedia API and REST Summary endpoints.
    Fetches official introductory lead paragraphs for verified factual correctness.
    """
    headers = {"User-Agent": "APEXResearchBot/1.0 (https://github.com; contact@apex-bot.org)"}
    api_url = "https://en.wikipedia.org/w/api.php"
    cleaned_q = extract_search_terms(query)

    search_items = []
    # 1. Search with natural question query first (CirrusSearch optimizes for natural language questions)
    for q_try in [query.strip(), cleaned_q]:
        if not q_try:
            continue
        params = {
            "action": "query",
            "list": "search",
            "srsearch": q_try,
            "format": "json",
            "srlimit": max_results,
            "utf8": 1
        }
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                items = resp.json().get("query", {}).get("search", [])
                for item in items:
                    if not any(it.get("title") == item.get("title") for it in search_items):
                        search_items.append(item)
            if len(search_items) >= max_results:
                break
        except Exception as e:
            logger.debug(f"Wikipedia search query error for '{q_try}': {e}")

    results = []
    for item in search_items[:max_results]:
        title = item.get("title", "")
        if not title:
            continue

        raw_snippet = item.get("snippet", "")
        clean_snip = clean_text(re.sub(r'<[^>]+>', '', raw_snippet))
        encoded_title = urllib.parse.quote(title.replace(' ', '_'))
        page_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

        # 2. Enrich with full lead paragraph summary from official REST API
        parts = []
        if clean_snip:
            parts.append(clean_snip)
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        try:
            s_resp = requests.get(summary_url, headers=headers, timeout=2.5)
            if s_resp.status_code == 200:
                extract = clean_text(s_resp.json().get("extract", ""))
                if extract and clean_snip.lower() not in extract.lower():
                    parts.append(extract[:350])
        except Exception:
            pass

        best_text = " ... ".join(parts) if parts else clean_snip

        if best_text:
            results.append({
                "title": f"{title} - Wikipedia",
                "snippet": best_text,
                "url": page_url,
                "domain": "en.wikipedia.org"
            })

    return results


def search_web(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """
    Unified multi-source web search with LRU caching.
    Retrieves verified real-time web facts from DuckDuckGo and Wikipedia.
    """
    cache_key = query.strip().lower()
    now = time.time()
    
    # 1. Check in-memory cache
    if cache_key in _WEB_CACHE:
        cached_time, cached_res = _WEB_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_res

    # 2. Execute DuckDuckGo live search
    results = search_duckduckgo(query, max_results=max_results)

    # 3. If DDG results are sparse or unavailable, supplement with Wikipedia
    if len(results) < 2:
        wiki_res = search_wikipedia(query, max_results=max_results - len(results))
        for wr in wiki_res:
            if not any(wr['url'] == r['url'] for r in results):
                results.append(wr)

    # 4. Save to cache
    if results:
        _WEB_CACHE[cache_key] = (now, results)

    return results


def needs_web_search(query: str,
                     has_doc_intent: bool = False,
                     has_memory_intent: bool = False) -> bool:
    """
    Detect whether the query requires real-time web search for factual correctness.
    
    Returns True if:
      - The user explicitly requests web / internet search.
      - Or the query is a factual / informational question that does NOT match local PDF documents
        and does NOT match episodic conversation memory.
    """
    q = query.strip()
    if not q:
        return False

    lower_q = q.lower()

    # 1. If document intent was explicitly detected, local PDF documents take precedence
    if has_doc_intent:
        return False

    # 2. If user is asking about episodic memory, prioritize episodic conversation memory
    if has_memory_intent:
        return False

    # Check for memory words (e.g. "do you remember", "my dog", "my name")
    if any(m in lower_q for m in MEMORY_INDICATORS):
        return False

    # 3. Explicit web search requests ALWAYS trigger web search
    if EXPLICIT_WEB_REGEX.search(q):
        return True

    # 4. Casual chitchat / greetings / creative generation should NOT trigger web search
    if CHITCHAT_REGEX.search(q):
        return False

    # 5. Check if query matches factual / world knowledge patterns
    if FACTUAL_QUERY_REGEX.search(q):
        return True

    # 6. Informational questions containing question words and substantial length
    words = lower_q.split()
    if len(words) >= 4 and any(w in words for w in ("who", "what", "where", "when", "why", "how", "which", "capital", "population", "release", "current", "latest")):
        return True

    return False
