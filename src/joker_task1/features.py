from __future__ import annotations

from collections import Counter

from .retriever import HybridTask1Retriever


PUNCT = set("!?\"'`.,:;-")


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def humor_features(query: str, doc_text: str) -> dict[str, float]:
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
    return {
        "exact_match": 1.0 if q_lower and q_lower in d_lower else 0.0,
        "token_overlap": _safe_div(overlap, len(q_tokens)),
        "char_overlap": _safe_div(gram_overlap, len(q_grams)),
        "doc_len_norm": min(len(d_tokens) / 40.0, 1.0),
        "punct_norm": min(punct_count / 10.0, 1.0),
        "exclaim_norm": min(exclaim_count / 3.0, 1.0),
        "quote_norm": min(quote_count / 4.0, 1.0),
        "repeated_words_norm": min(repeated_words / 4.0, 1.0),
    }
