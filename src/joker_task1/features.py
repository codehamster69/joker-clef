from __future__ import annotations

from collections import Counter
from functools import lru_cache

from .retriever import HybridTask1Retriever


PUNCT = set("!?\"'`.,:;-")

# Simple positive/negative word sets for sentiment-contrast detection.
_POS_WORDS = {"good", "great", "love", "happy", "joy", "nice", "wonderful", "best",
              "amazing", "beautiful", "awesome", "excellent", "perfect", "brilliant"}
_NEG_WORDS = {"bad", "hate", "ugly", "terrible", "horrible", "worst", "awful",
              "disgusting", "dreadful", "stupid", "idiot", "dead", "death", "kill",
              "pain", "miserable", "sad", "cry", "poor", "broke"}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


@lru_cache(maxsize=4096)
def _word_polysemy(word: str) -> int:
    """Number of WordNet synsets for a word (cached). Returns 0 on import failure."""
    try:
        from nltk.corpus import wordnet
        return len(wordnet.synsets(word))
    except Exception:
        return 0


def humor_features(query: str, doc_text: str,
                   idf_map: dict[str, float] | None = None) -> dict[str, float]:
    q_lower = query.lower()
    d_lower = doc_text.lower()
    q_tokens = HybridTask1Retriever.tokenize(query)
    d_tokens = HybridTask1Retriever.tokenize(doc_text)
    q_counter = Counter(q_tokens)
    d_counter = Counter(d_tokens)
    overlap = sum(min(q_counter[t], d_counter[t]) for t in q_counter)
    q_grams = Counter(HybridTask1Retriever.char_ngrams(q_lower))
    d_grams = Counter(HybridTask1Retriever.char_ngrams(d_lower))
    gram_overlap = sum(min(q_grams[g], d_grams[g]) for g in q_grams)
    punct_count = sum(1 for ch in doc_text if ch in PUNCT)
    exclaim_count = doc_text.count("!")
    quote_count = doc_text.count('"') + doc_text.count("“") + doc_text.count("”")
    repeated_words = sum(1 for _, c in Counter(d_tokens).items() if c >= 2)

    # --- New feature 1: query polysemy (ambiguous queries → more pun potential) ---
    synset_counts = [_word_polysemy(t) for t in q_tokens if len(t) > 2]
    query_polysemy = min(_safe_div(sum(synset_counts), max(len(synset_counts), 1)) / 10.0, 1.0)

    # --- New feature 2: average word rarity in document (rare words = funnier) ---
    if idf_map and d_tokens:
        avg_idf = sum(idf_map.get(t, 0.0) for t in d_tokens) / len(d_tokens)
        max_idf = max(idf_map.values()) if idf_map else 1.0
        avg_word_rarity = min(avg_idf / max(max_idf, 1e-9), 1.0)
    else:
        avg_word_rarity = 0.0

    # --- New feature 3: question mark in doc (setup-punchline pattern) ---
    question_in_doc = 1.0 if "?" in doc_text else 0.0

    # --- New feature 4: sentiment contrast (incongruity = humor signal) ---
    d_word_set = set(d_tokens)
    has_pos = bool(d_word_set & _POS_WORDS)
    has_neg = bool(d_word_set & _NEG_WORDS)
    doc_sentiment_contrast = 1.0 if (has_pos and has_neg) else 0.0

    return {
        "exact_match": 1.0 if q_lower and q_lower in d_lower else 0.0,
        "token_overlap": _safe_div(overlap, len(q_tokens)),
        "char_overlap": _safe_div(gram_overlap, len(q_grams)),
        "doc_len_norm": min(len(d_tokens) / 40.0, 1.0),
        "punct_norm": min(punct_count / 10.0, 1.0),
        "exclaim_norm": min(exclaim_count / 3.0, 1.0),
        "quote_norm": min(quote_count / 4.0, 1.0),
        "repeated_words_norm": min(repeated_words / 4.0, 1.0),
        "query_polysemy": query_polysemy,
        "avg_word_rarity": avg_word_rarity,
        "question_in_doc": question_in_doc,
        "doc_sentiment_contrast": doc_sentiment_contrast,
    }
