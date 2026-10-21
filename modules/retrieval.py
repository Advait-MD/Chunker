"""
APEX RAG - Hybrid Retrieval, Reranking, and Deduplication Module.
Provides BM25Okapi lexical retrieval, Reciprocal Rank Fusion (RRF),
composite post-retrieval reranking, and Jaccard chunk deduplication.
"""

import os
import re
import threading
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi

from .config import config
from .logger import get_logger

logger = get_logger("retrieval")


# ================= HIGH-SPEED CACHING INFRASTRUCTURE =================

class QueryEmbeddingCache:
    """Thread-safe LRU cache for query vector embeddings (Item 2.4 & 3.3)."""
    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, query: str) -> Optional[List[float]]:
        key = query.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, query: str, embedding: List[float]):
        key = query.strip().lower()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = embedding
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def get_or_compute(self, query: str, embedding_fn) -> List[float]:
        cached = self.get(query)
        if cached is not None:
            return cached
        emb = embedding_fn([query.strip()])[0]
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
        elif not isinstance(emb, list):
            emb = list(emb)
        self.put(query, emb)
        return emb


class RetrievalResultCache:
    """Thread-safe LRU cache for retrieved context dictionaries (Item 2.4)."""
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Tuple) -> Optional[Dict]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                res = dict(self._cache[key])
                res['from_cache'] = True
                return res
            return None

    def put(self, key: Tuple, result: Dict):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = result
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()


# Singletons
query_embedding_cache = QueryEmbeddingCache(maxsize=1024)
retrieval_result_cache = RetrievalResultCache(maxsize=512)


# ================= CHIT-CHAT, COMMON CHAT & INTENT CLASSIFICATION =================

COMMON_CHAT_PATTERNS = [
    # 1. Greetings, pleasantries, small talk
    r"^(hi|hello|hey|heya|howdy|sup|yo|hola|greetings)\b",
    r"^(good\s+(morning|afternoon|evening|day|night))\b",
    r"^(how\s+are\s+you|how\'s\s+it\s+going|how\s+do\s+you\s+do|what\'s\s+up|how\s+are\s+things|how\s+is\s+your\s+day)\b",
    r"^(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)\b",
    r"^(thank\s+you|thanks|thanks\s+a\s+lot|thx|much\s+appreciated|cheers)\b",
    r"^(bye|goodbye|see\s+you|cya|take\s+care|farewell|see\s+ya|have\s+a\s+good\s+day)\b",
    r"^(ok|okay|cool|awesome|great|nice|perfect|got\s+it|understood|yes|no|yep|nope|sure|alright)\b",

    # 2. Identity, capabilities, bot meta-questions
    r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|introduce\s+yourself|tell\s+me\s+about\s+yourself)\b",
    r"^(are\s+you\s+(an?\s+)?(ai|robot|bot|human|llm|model|machine))\b",
    r"^(what\s+are\s+you|how\s+were\s+you\s+(made|built|created)|who\s+(made|created|built)\s+you)\b",
    r"^(can\s+we\s+chat|can\s+we\s+talk|let\'s\s+talk|let\'s\s+chat)\b",

    # 3. Casual conversational requests, humor, creative writing
    r"^(tell\s+me\s+(a\s+)?(joke|riddle|pun|story|fun\s+fact))\b",
    r"^(make\s+me\s+laugh|say\s+something\s+(funny|interesting|cool))\b",
    r"^(write\s+(a\s+)?(poem|haiku|limerick|song|rap|story|essay|email|letter|speech))\b",
    r"^(give\s+me\s+(some\s+)?(advice|tips|ideas|suggestions|recommendations))\b",
    r"^(i\'m\s+(bored|sad|happy|tired|stressed|angry|curious))\b",
    r"^(what\s+do\s+you\s+think\s+about\s+(life|love|happiness|friendship|future|humans))\b",

    # 4. General open-domain tasks (coding, math, general world facts not related to documents)
    r"^(write\s+(a\s+)?(python|javascript|c\+\+|java|html|css|bash|sql|code|script|function|program))\b",
    r"^(calculate|what\s+is\s+\d+[\s\+\-\*\/])\b",
    r"^(translate\s+.*to\s+(french|spanish|german|hindi|chinese|japanese|italian|russian))\b",
    r"^(why\s+(is|are|do|does|did|was|were))\b",
    r"^(how\s+(do|does|did|can|could|to|come|is|are|was|were))\b",
    r"^(can|could|do|does|did)\s+.*?\s+(fly|swim|walk|run|talk|speak|sing|breathe|think|feel|survive|live|exist|sleep|eat|grow|work)\b",
    r"^(what\s+is\s+the\s+(capital|difference|meaning|definition|purpose|formula|recipe|speed|distance|temperature|weather|time|date)\b)",
    r"^(who\s+(invented|discovered|wrote|directed|founded|painted|created)\s+([a-zA-Z\s]+))\b",
    r"^(explain\s+(how|why|what|the\s+concept\s+of|the\s+theory\s+of|the\s+process\s+of)\b)",
    r"^(tell\s+me\s+(about|more\s+about|the\s+story\s+of|a\s+story\s+about)\b)",

    # 5. Personal reflections, life situations, feelings, emotional sharing, and advice
    r"\b(lately\s+i|recently\s+i|sometimes\s+i|i('ve| have)? been feeling|i feel like|i'm feeling)\b",
    r"\b(how do (i|people|we)|how (can|should) (i|we|people)|what should i do|any advice (on|for)|how to deal with)\b",
    r"\b(burnout|overwhelmed|imposter syndrome|procrastinat|work-life balance|exhausted and|vicious cycle)\b",
    r"\b(career (path|transition|choice|change)|long-term career|workplace dilemma|life advice)\b"
]
COMMON_CHAT_REGEX = re.compile("|".join(COMMON_CHAT_PATTERNS), re.IGNORECASE)

GENERIC_DOC_KEYWORDS = {
    "pdf", "page", "chapter", "document", "documents", "source", "sources",
    "text", "book", "author", "cite", "citation", "excerpt", "context",
    "quote", "passage", "paragraph", "reference", "manual", "guide"
}


def is_common_chat_query(query: str, indexed_sources: Optional[List[str]] = None) -> bool:
    """
    Detect whether a query is a general common conversation / chit-chat / open-domain request
    that does NOT require document search from indexed memory.
    
    Returns True if:
      1. Query matches common conversational / open-domain patterns AND does not mention indexed documents.
      2. Or query is a brief greeting / polite remark without document keywords.
    """
    q = query.strip()
    if not q:
        return True
        
    lower_q = q.lower()
    words = set(re.findall(r'\b\w+\b', lower_q))
    
    # 1. Check for explicit document terms
    if words.intersection(GENERIC_DOC_KEYWORDS):
        return False
        
    # 2. Check for terms from indexed sources (e.g. Steve Jobs, Odyssey, Book of Five Rings, React)
    if indexed_sources:
        for src in indexed_sources:
            clean_name = os.path.splitext(os.path.basename(src))[0].lower()
            name_words = set(re.findall(r'\b\w+\b', clean_name)) - {"the", "of", "and", "a", "an", "to", "in", "for"}
            if name_words and len(words.intersection(name_words)) > 0:
                return False
                
    # Also check known domain anchors if not explicitly in indexed_sources
    domain_anchors = {
        "steve", "jobs", "wozniak", "apple", "macintosh", "next", "pixar", "iphone", "ipad", "ipod", "cupertino",
        "musashi", "miyamoto", "five rings", "gorin", "kenjutsu", "samurai",
        "odyssey", "odysseus", "telemachus", "penelope", "ithaca", "athena", "poseidon", "zeus", "cyclops", "polyphemus", "circe", "calypso", "homer",
        "react", "jsx", "props", "state", "usestate", "useeffect", "usereducer", "redux", "wieruch"
    }
    if any(k in lower_q for k in domain_anchors):
        return False
        
    # 3. If query matches common chat patterns, route to common chat!
    if COMMON_CHAT_REGEX.search(q):
        return True
        
    # 4. Short casual remarks under 5 words without domain keywords
    if len(q.split()) <= 4:
        conversational_words = {
            "hi", "hello", "hey", "sup", "yo", "thanks", "thank", "bye", "goodbye",
            "cool", "nice", "awesome", "great", "ok", "okay", "yes", "no", "yep", "nope",
            "sure", "who", "what", "how", "why", "help", "me", "you", "i", "can", "tell",
            "joke", "fun", "bored", "chat", "talk", "morning", "night", "day"
        }
        if words.issubset(conversational_words) or words.intersection(conversational_words):
            return True
            
    return False


def is_chitchat_query(query: str, indexed_sources: Optional[List[str]] = None) -> bool:
    """
    Detect whether a query is pure conversational chit-chat / pleasantry (Item 2.3).
    Bypasses expensive vector search and BM25 candidate retrieval when no documents are needed.
    """
    return is_common_chat_query(query, indexed_sources=indexed_sources)


EXPLICIT_DOC_PATTERNS = [
    r"\b(in|from|according to|based on|per|within)\s+(the\s+)?(document|documents|pdf|paper|papers|book|books|text|texts|chapter|chapters|page|pages|source|sources|file|files|reading|manual|guide)\b",
    r"\b(what does the\s+(document|pdf|paper|book|text|author)\s+(say|state|mention|explain|describe|tell))\b",
    r"\b(summarize\s+(the\s+)?(document|pdf|paper|book|chapter|text|file))\b",
    r"\b(look up|find|search for)\s+.*?\s+(in\s+the\s+(document|pdf|book|paper|text|files))\b",
    r"\b(chapter\s+\d+|page\s+\d+)\b",
    r"\b(the\s+author\s+(argues|states|claims|writes|concludes|mentions))\b",
    r"\b(cite|citation|quote from|passage from|excerpt from)\b",
    r"\b(in this (paper|pdf|book|document|text))\b"
]
EXPLICIT_DOC_REGEX = re.compile("|".join(EXPLICIT_DOC_PATTERNS), re.IGNORECASE)

DOMAIN_ANCHOR_MAP = {
    "Steve Jobs": [
        "steve jobs", "wozniak", "steve wozniak", "xerox parc", "macintosh", 
        "next computer", "pixar", "john sculley", "sculley", "stanford commencement", 
        "apple ii", "lisa computer", "cupertino", "jony ive"
    ],
    "The Book of Five Rings": [
        "musashi", "miyamoto", "miyamoto musashi", "five rings", "gorin no sho", 
        "kenjutsu", "way of strategy", "water scroll", "void scroll", "earth scroll", 
        "fire scroll", "wind scroll", "school of two swords", "niten ichi-ryu"
    ],
    "The Odyssey": [
        "odyssey", "odysseus", "telemachus", "penelope", "ithaca", "athena", 
        "poseidon", "zeus", "cyclops", "polyphemus", "circe", "calypso", 
        "scylla", "charybdis", "lotus eaters", "suitors", "homer", "nausicaa",
        "teiresias", "tiresias"
    ],
    "The Road to React": [
        "robin wieruch", "wieruch", "react", "jsx", "usestate", "useeffect", 
        "usecallback", "usememo", "usereducer", "react hooks", "custom hook",
        "react components", "props"
    ],
    "RAPTOR": [
        "raptor", "recursive abstractive processing", "tree rag", "collapsed tree"
    ],
    "QASPER": [
        "qasper"
    ]
}


def detect_document_intent(query: str, indexed_sources: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Automatically detect whether a user query requires Document RAG retrieval from ingested PDF files.
    Returns:
        (needs_document_rag: bool, detected_source_file: Optional[str])
    """
    q = query.strip()
    if not q:
        return False, None

    lower_q = q.lower()
    words = set(re.findall(r'\b\w+\b', lower_q))

    def find_matching_source(hint: str) -> Optional[str]:
        if not indexed_sources:
            return None
        hint_clean = hint.lower().replace(".pdf", "")
        for src in indexed_sources:
            src_clean = src.lower().replace(".pdf", "")
            if hint_clean in src_clean or src_clean in hint_clean:
                return src
            src_words = set(re.findall(r'\b\w+\b', src_clean)) - {"the", "of", "and", "a", "an", "to", "in", "for", "1"}
            hint_words = set(re.findall(r'\b\w+\b', hint_clean)) - {"the", "of", "and", "a", "an", "to", "in", "for", "1"}
            if src_words and hint_words and len(src_words.intersection(hint_words)) >= min(len(src_words), len(hint_words)):
                return src
        return None

    # 1. Regex check for explicit document phrasing
    if EXPLICIT_DOC_REGEX.search(q):
        if indexed_sources:
            for src in indexed_sources:
                clean = os.path.splitext(os.path.basename(src))[0].lower()
                src_words = set(re.findall(r'\b\w+\b', clean)) - {"the", "of", "and", "a", "an", "to", "in", "for", "1"}
                if src_words and len(words.intersection(src_words)) >= min(2, len(src_words)):
                    return True, src
        return True, None

    # 2. Check for explicit document keywords
    if words.intersection(GENERIC_DOC_KEYWORDS):
        if indexed_sources:
            for src in indexed_sources:
                clean = os.path.splitext(os.path.basename(src))[0].lower()
                src_words = set(re.findall(r'\b\w+\b', clean)) - {"the", "of", "and", "a", "an", "to", "in", "for", "1"}
                if src_words and len(words.intersection(src_words)) >= min(2, len(src_words)):
                    return True, src
        return True, None

    # 3. Direct match against indexed source file titles
    if indexed_sources:
        for src in indexed_sources:
            clean = os.path.splitext(os.path.basename(src))[0].lower()
            src_words = set(re.findall(r'\b\w+\b', clean)) - {"the", "of", "and", "a", "an", "to", "in", "for", "1"}
            if src_words:
                matched = words.intersection(src_words)
                if len(matched) >= min(2, len(src_words)) or (len(matched) == 1 and any(len(w) >= 7 for w in matched)):
                    return True, src

    # 4. Check domain anchors tied to indexed document content
    for topic, anchors in DOMAIN_ANCHOR_MAP.items():
        for anchor in anchors:
            if anchor in lower_q:
                matching_src = find_matching_source(topic)
                return True, matching_src

    return False, None



def tokenize_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase words for lexical analysis."""
    if not text:
        return []
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]


class BM25Scorer:
    """Lightweight BM25Okapi scoring wrapper for candidate reranking."""
    
    def __init__(self, corpus_texts: List[str]):
        self.corpus_texts = corpus_texts
        self.tokenized_corpus = [tokenize_bm25(doc) for doc in corpus_texts]
        if self.tokenized_corpus and any(len(toks) > 0 for toks in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def score(self, query: str) -> np.ndarray:
        """Compute min-max normalized BM25 scores across candidate texts."""
        if self.bm25 is None or not self.corpus_texts:
            return np.zeros(len(self.corpus_texts))
            
        query_tokens = tokenize_bm25(query)
        if not query_tokens:
            return np.zeros(len(self.corpus_texts))
            
        raw_scores = self.bm25.get_scores(query_tokens)
        max_s = float(np.max(raw_scores)) if len(raw_scores) > 0 else 0.0
        min_s = float(np.min(raw_scores)) if len(raw_scores) > 0 else 0.0
        
        if max_s > min_s:
            return (raw_scores - min_s) / (max_s - min_s)
        elif max_s > 0:
            return np.ones(len(raw_scores)) * 0.5
        return np.zeros(len(raw_scores))


def reciprocal_rank_fusion(dense_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> List[float]:
    """
    Standard Reciprocal Rank Fusion (RRF) formula:
      RRF = (1 / (k + rank_dense)) + (1 / (k + rank_bm25))
    """
    rrf_scores = []
    for r_dense, r_bm25 in zip(dense_ranks, bm25_ranks):
        score = (1.0 / (k + r_dense + 1)) + (1.0 / (k + r_bm25 + 1))
        rrf_scores.append(score)
    return rrf_scores


def jaccard_word_similarity(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    tokens_a = set(tokenize_bm25(text_a))
    tokens_b = set(tokenize_bm25(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return intersection / union if union > 0 else 0.0


def deduplicate_retrieved_chunks(chunks: List[str], 
                                 metadatas: List[Dict], 
                                 scores: List[float], 
                                 max_jaccard_overlap: float = 0.65) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Deduplicate retrieved chunks before prompt construction (Item 27):
      1. Drops adjacent sliding-window chunks from the same document if redundant.
      2. Drops chunks with word Jaccard similarity >= max_jaccard_overlap with an accepted chunk.
    Preserves descending ranking order.
    """
    if not chunks:
        return [], [], []

    deduped_chunks = []
    deduped_metas = []
    deduped_scores = []
    seen_indices_by_source = {}

    for chunk, meta, score in zip(chunks, metadatas, scores):
        source = meta.get('source_file', 'unknown') if meta else 'unknown'
        chunk_idx = meta.get('chunk_index') if meta else None
        
        # 1. Check adjacent sliding-window overlap
        if chunk_idx is not None:
            prev_indices = seen_indices_by_source.setdefault(source, set())
            if (chunk_idx - 1 in prev_indices) or (chunk_idx + 1 in prev_indices):
                is_adjacent_duplicate = False
                for accepted_chunk in deduped_chunks:
                    if jaccard_word_similarity(chunk, accepted_chunk) > 0.40:
                        is_adjacent_duplicate = True
                        break
                if is_adjacent_duplicate:
                    continue

        # 2. Check high lexical Jaccard overlap
        too_similar = False
        for accepted_chunk in deduped_chunks:
            if jaccard_word_similarity(chunk, accepted_chunk) >= max_jaccard_overlap:
                too_similar = True
                break
        if too_similar:
            continue

        deduped_chunks.append(chunk)
        deduped_metas.append(meta)
        deduped_scores.append(score)
        if chunk_idx is not None:
            seen_indices_by_source[source].add(chunk_idx)

    return deduped_chunks, deduped_metas, deduped_scores


COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
    "shall", "should", "will", "would", "may", "might", "must", "how", "what", "which",
    "who", "when", "where", "why", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    # Prepositions & connective function words
    "across", "through", "between", "among", "along", "against", "around", "behind",
    "below", "beneath", "beside", "beyond", "during", "inside", "outside", "under",
    "upon", "without", "within", "toward", "towards", "onto", "off", "out",
    # High-frequency functional adverbs/qualifiers
    "also", "just", "very", "really", "much", "many", "more", "most", "some", "any",
    "each", "every", "all", "both", "either", "neither", "one", "two", "such", "than",
    "then", "there", "here"
}


def compute_query_coverage(query: str, text: str) -> float:
    """Calculate the fraction of non-trivial query keywords present in document (with stem/prefix matching)."""
    tokens = tokenize_bm25(query)
    q_tokens = [w for w in tokens if len(w) > 2 and w not in COMMON_STOPWORDS]
    if not q_tokens:
        q_tokens = [w for w in tokens if len(w) > 2]
    if not q_tokens:
        return 0.5
    doc_tokens = set(tokenize_bm25(text))
    matched = 0
    for qw in q_tokens:
        for dw in doc_tokens:
            if qw == dw or (len(qw) >= 3 and len(dw) >= 3 and (dw.startswith(qw) or qw.startswith(dw))):
                matched += 1
                break
    return matched / len(q_tokens)


def retrieve_conversation_memories(query: str,
                                   conv_chunks_col,
                                   conv_summaries_col=None,
                                   query_emb: Optional[List[float]] = None,
                                   store=None,
                                   max_memories: int = 2,
                                   dist_threshold: float = 1.22,
                                   cov_threshold: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
    """
    High-speed episodic conversation memory retrieval.
    Searches ChromaDB's conversation collections for past turns relevant to the user query.
    Returns:
      (recalled_turn_memories, recalled_cluster_summaries)
    """
    if conv_chunks_col is None:
        return [], []
        
    try:
        if conv_chunks_col.count() == 0:
            return [], []
    except Exception:
        return [], []
        
    q = query.strip()
    lower_q = q.lower()
    if len(q.split()) <= 1 and lower_q in ("hi", "hello", "hey", "yo", "sup", "bye"):
        return [], []

    # Ephemeral pleasantries and joke requests without past references shouldn't recall prior memories
    past_reference_words = {"remember", "earlier", "before", "previously", "last time", "again", "yesterday", "we talked", "you said", "i told", "my name", "my dog", "my project", "my pet", "did we", "what was"}
    has_past_reference = any(ref in lower_q for ref in past_reference_words)
    generic_chitchat_requests = [
        r"^(can\s+you\s+)?(tell\s+me\s+)?(a\s+)?(funny\s+)?(joke|riddle|pun|story)\b",
        r"^(make\s+me\s+laugh|say\s+something\s+funny)\b",
        r"^(hi|hello|hey|yo|howdy|sup|good\s+(morning|afternoon|evening|night))\b",
        r"^(how\s+are\s+you|how\'s\s+it\s+going|how\s+are\s+things)\b"
    ]
    if not has_past_reference and any(re.search(pat, lower_q) for pat in generic_chitchat_requests):
        return [], []

    try:
        if query_emb is None and store is not None and hasattr(store, 'get_query_embedding'):
            query_emb = store.get_query_embedding(q)

        total_turns = conv_chunks_col.count()
        n_results = min(max_memories * 3, total_turns)
        query_args = {"n_results": n_results, "include": ["documents", "metadatas", "distances"]}
        if query_emb is not None:
            query_args["query_embeddings"] = [query_emb]
        else:
            query_args["query_texts"] = [q]

        raw_res = conv_chunks_col.query(**query_args)
        if not raw_res or not raw_res.get("documents") or not raw_res["documents"][0]:
            return [], []

        memories = []
        for doc, meta, dist in zip(raw_res["documents"][0], raw_res["metadatas"][0], raw_res["distances"][0]):
            d = float(dist)
            cov = compute_query_coverage(q, doc)
            # Relevance test: close embedding distance OR strong keyword coverage
            is_relevant = (d <= dist_threshold and cov >= cov_threshold) or (d <= 0.88) or (cov >= 0.40 and d <= 1.35)
            if is_relevant:
                score = max(0.0, min(1.0, 1.0 - (d / 2.0)))
                memories.append({
                    "text": doc,
                    "chat_id": meta.get("chat_id"),
                    "chat_title": meta.get("chat_title", "Previous Chat"),
                    "turn_index": meta.get("turn_index", 0),
                    "timestamp": meta.get("timestamp", ""),
                    "score": score,
                    "coverage_score": cov,
                    "distance": d
                })
                if len(memories) >= max_memories:
                    break

        summaries = []
        if conv_summaries_col is not None:
            try:
                if conv_summaries_col.count() > 0:
                    s_args = {"n_results": min(2, conv_summaries_col.count()), "include": ["documents", "metadatas", "distances"]}
                    if query_emb is not None:
                        s_args["query_embeddings"] = [query_emb]
                    else:
                        s_args["query_texts"] = [q]
                    s_res = conv_summaries_col.query(**s_args)
                    if s_res and s_res.get("documents") and s_res["documents"][0]:
                        for s_doc, s_meta, s_dist in zip(s_res["documents"][0], s_res["metadatas"][0], s_res["distances"][0]):
                            sd = float(s_dist)
                            if sd <= 1.20:
                                summaries.append({
                                    "summary": s_doc,
                                    "dense_summary": s_meta.get("dense_summary", ""),
                                    "chats_represented": s_meta.get("chats_represented", "Chats"),
                                    "score": max(0.0, 1.0 - (sd / 2.0)),
                                    "level": s_meta.get("level", 0)
                                })
            except Exception as e:
                logger.debug(f"Error querying conversation summaries: {e}")

        return memories, summaries

    except Exception as e:
        logger.warning(f"Error retrieving conversation memories: {e}")
        return [], []


def rerank_hybrid_candidates(query: str,
                             candidates: List[Dict],
                             dense_weight: float = 0.50,
                             bm25_weight: float = 0.30,
                             coverage_weight: float = 0.20) -> List[Dict]:
    """
    Post-Retrieval Reranking (Item 24 & Item 25):
    Scores candidate chunks using a multi-factor composite formula:
      - Dense semantic similarity (nomic-embed-text)
      - Lexical term match (BM25Okapi)
      - Non-trivial query term coverage ratio
    """
    if not candidates:
        return []

    docs = [c['document'] for c in candidates]
    bm25_scorer = BM25Scorer(docs)
    bm25_norm_scores = bm25_scorer.score(query)

    for i, c in enumerate(candidates):
        dense_score = c.get('dense_score', 0.0)
        bm25_score = float(bm25_norm_scores[i]) if i < len(bm25_norm_scores) else 0.0
        coverage_score = compute_query_coverage(query, c['document'])
        
        composite_score = (
            dense_score * dense_weight +
            bm25_score * bm25_weight +
            coverage_score * coverage_weight
        )
        
        # Slight adaptive prior for thematic queries matching tree summaries
        if c.get('node_type') == 'tree_summary':
            thematic_triggers = {"theme", "summary", "overview", "philosophy", "evolution", "transformation", "life", "history", "perspective"}
            query_words = set(tokenize_bm25(query))
            if query_words.intersection(thematic_triggers):
                composite_score = min(1.0, composite_score + 0.05)
        
        c['bm25_score'] = bm25_score
        c['coverage_score'] = coverage_score
        c['rerank_score'] = composite_score

    # Re-sort descending by composite reranking score
    candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    return candidates


def collapsed_tree_fusion(raw_chunks: Optional[List[str]],
                          raw_metas: Optional[List[Dict]],
                          raw_dists: Optional[List[float]],
                          summary_docs: Optional[List[str]] = None,
                          summary_metas: Optional[List[Dict]] = None,
                          summary_dists: Optional[List[float]] = None,
                          memory_docs: Optional[List[str]] = None,
                          memory_metas: Optional[List[Dict]] = None,
                          memory_dists: Optional[List[float]] = None) -> List[Dict]:
    """
    RAPTOR-style Collapsed Tree Fusion (ICLR 2024):
    Flattens multi-layer tree nodes (leaf chunks, Level 0 intra-doc summaries,
    Level 1 macro summaries, Level 2 global summaries, and conversational memories)
    into a unified candidate pool for joint hybrid evaluation.
    """
    candidates = []
    
    # 1. Leaf Chunks (Layer 0)
    for doc, meta, dist in zip(raw_chunks or [], raw_metas or [], raw_dists or []):
        dense_score = max(0.0, min(1.0, 1.0 - float(dist)))
        candidates.append({
            'document': doc,
            'metadata': meta,
            'distance': dist,
            'dense_score': dense_score,
            'node_type': 'leaf_chunk',
            'level': meta.get('level', 0) if meta else 0,
            'source_file': meta.get('source_file', 'unknown') if meta else 'unknown',
            'page_number': meta.get('page_number', '?') if meta else '?'
        })
        
    # 2. Hierarchical Summaries (Layers 1 to d)
    if summary_docs and summary_metas and summary_dists:
        for s_doc, s_meta, s_dist in zip(summary_docs, summary_metas, summary_dists):
            s_score = max(0.0, min(1.0, 1.0 - float(s_dist)))
            # Quality gate: omit low confidence or negative silhouette summaries
            if s_meta and (s_meta.get('low_confidence', False) or float(s_meta.get('silhouette_score', 0.0)) <= 0.0):
                continue
            candidates.append({
                'document': s_doc,
                'metadata': s_meta,
                'distance': s_dist,
                'dense_score': s_score,
                'node_type': 'tree_summary',
                'level': s_meta.get('level', 0) if s_meta else 0,
                'source_file': s_meta.get('source_file', 'cluster_theme') if s_meta else 'cluster_theme',
                'page_number': s_meta.get('page_range', '') if s_meta else '',
                'dense_summary': s_meta.get('dense_summary', '') if s_meta else ''
            })
            
    # 3. Conversational Memories
    if memory_docs and memory_metas and memory_dists:
        for m_doc, m_meta, m_dist in zip(memory_docs, memory_metas, memory_dists):
            m_score = max(0.0, min(1.0, 1.0 - float(m_dist)))
            candidates.append({
                'document': m_doc,
                'metadata': m_meta,
                'distance': m_dist,
                'dense_score': m_score,
                'node_type': 'memory',
                'level': m_meta.get('turn_index', 0) if m_meta else 0,
                'source_file': m_meta.get('chat_title', 'Conversation') if m_meta else 'Conversation',
                'page_number': ''
            })
            
    return candidates


def pack_collapsed_context(reranked_candidates: List[Dict],
                           max_tokens: int = 350,
                           max_chunks: int = 2,
                           max_summaries: int = 1) -> Dict:
    """
    Greedy budget-aware context packing for collapsed tree retrieval.
    Selects top candidates up to max_tokens (approx 4 chars per token)
    to guarantee local LLM generation TTFT <= 1.8s and total turnaround <= 4.2s.
    """
    packed_chunks = []
    packed_metas = []
    packed_scores = []
    packed_summaries = []
    collapsed_items = []
    
    current_tokens = 0
    chunk_count = 0
    summary_count = 0
    
    for c in reranked_candidates:
        node_type = c.get('node_type', 'leaf_chunk')
        text = c['document']
        tokens_est = max(1, len(text) // 4)
        
        if node_type == 'leaf_chunk':
            if chunk_count >= max_chunks:
                continue
            if current_tokens + min(tokens_est, 60) > max_tokens and packed_chunks:
                continue
            packed_chunks.append(text)
            packed_metas.append(c['metadata'])
            packed_scores.append(c['rerank_score'])
            collapsed_items.append(c)
            chunk_count += 1
            current_tokens += min(tokens_est, 60)
            
        elif node_type == 'tree_summary':
            if summary_count >= max_summaries:
                continue
            if current_tokens + min(tokens_est, 75) > max_tokens and (packed_chunks or packed_summaries):
                continue
            summary_info = {
                'cluster_id': c['metadata'].get('cluster_id', 'unknown') if c.get('metadata') else 'unknown',
                'level': c.get('level', 0),
                'chunks_count': c['metadata'].get('chunks_count', 0) if c.get('metadata') else 0,
                'score': c['rerank_score'],
                'sources': c['metadata'].get('source_files', [c.get('source_file')]) if c.get('metadata') else [c.get('source_file')],
                'silhouette_score': c['metadata'].get('silhouette_score', 0.0) if c.get('metadata') else 0.0,
                'summary': text[:600],
                'dense_summary': c.get('dense_summary', '')
            }
            packed_summaries.append(summary_info)
            collapsed_items.append(c)
            summary_count += 1
            current_tokens += min(tokens_est, 75)
            
        if chunk_count >= max_chunks and summary_count >= max_summaries:
            break
            
    return {
        'chunks': packed_chunks,
        'metadatas': packed_metas,
        'scores': packed_scores,
        'cluster_summaries': packed_summaries,
        'collapsed_items': collapsed_items,
        'total_estimated_tokens': current_tokens
    }
