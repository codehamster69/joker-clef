from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class RetrievedDoc:
    docid: str
    score: float


class HybridTask1Retriever:
    """Efficient lexical retriever for JOKER Task 1.

    Hybrid score =
      bm25_weight * BM25(tokenized text)
      + char_weight * Character 3-5gram TF-IDF cosine
      + humor_weight * qrels prior
      + match_boost * exact substring match
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        bm25_weight: float = 1.0,
        char_weight: float = 0.35,
        humor_weight: float = 0.2,
        match_boost: float = 0.1,
    ):
        self.k1 = k1
        self.b = b
        self.bm25_weight = bm25_weight
        self.char_weight = char_weight
        self.humor_weight = humor_weight
        self.match_boost = match_boost

        self.doc_text_lower: dict[str, str] = {}
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_lens: dict[str, int] = {}
        self.df: defaultdict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.humor_prior: dict[str, float] = defaultdict(float)

        self.char_tf: dict[str, Counter[str]] = {}
        self.char_df: defaultdict[str, int] = defaultdict(int)
        self.char_idf: dict[str, float] = {}
        self.char_doc_norm: dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

    @staticmethod
    def char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
        t = re.sub(r"\s+", " ", text.lower()).strip()
        if not t:
            return []
        grams: list[str] = []
        for n in range(min_n, max_n + 1):
            if len(t) < n:
                continue
            grams.extend(t[i : i + n] for i in range(len(t) - n + 1))
        return grams

    def fit(self, docs: Iterable[dict], qrels: Iterable[dict] | None = None) -> None:
        docs = list(docs)
        total_len = 0

        for d in docs:
            docid = str(d["docid"])
            text = str(d["text"])
            text_lower = text.lower()
            self.doc_text_lower[docid] = text_lower

            tok = self.tokenize(text)
            tf = Counter(tok)
            self.term_freqs[docid] = tf
            self.doc_lens[docid] = len(tok)
            total_len += len(tok)
            for t in tf:
                self.df[t] += 1

            ctf = Counter(self.char_ngrams(text_lower))
            self.char_tf[docid] = ctf
            for g in ctf:
                self.char_df[g] += 1

        n_docs = max(1, len(self.term_freqs))
        self.avgdl = total_len / n_docs

        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        for gram, df in self.char_df.items():
            self.char_idf[gram] = math.log(1 + n_docs / (1 + df))

        for docid, ctf in self.char_tf.items():
            norm2 = 0.0
            for gram, tf in ctf.items():
                w = (1.0 + math.log(tf)) * self.char_idf.get(gram, 0.0)
                norm2 += w * w
            self.char_doc_norm[docid] = math.sqrt(norm2) if norm2 > 0 else 1.0

        if qrels:
            positive_ids = {str(r["docid"]) for r in qrels if int(r.get("qrel", 0)) > 0}
            for docid in self.term_freqs:
                self.humor_prior[docid] = 1.0 if docid in positive_ids else 0.0

    def bm25(self, query_tokens: list[str], docid: str) -> float:
        score = 0.0
        tf_doc = self.term_freqs[docid]
        dl = self.doc_lens[docid]
        norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        for t in query_tokens:
            tf = tf_doc.get(t, 0)
            if tf:
                idf = self.idf.get(t, 0.0)
                score += idf * (tf * (self.k1 + 1)) / (tf + norm)
        return score

    def char_tfidf_cosine(self, query_text: str, docid: str) -> float:
        qtf = Counter(self.char_ngrams(query_text))
        if not qtf:
            return 0.0

        dot = 0.0
        qnorm2 = 0.0
        doc_ctf = self.char_tf[docid]

        for gram, tfq in qtf.items():
            idf = self.char_idf.get(gram, 0.0)
            if idf == 0.0:
                continue
            qw = (1.0 + math.log(tfq)) * idf
            qnorm2 += qw * qw

            tfd = doc_ctf.get(gram, 0)
            if tfd:
                dw = (1.0 + math.log(tfd)) * idf
                dot += qw * dw

        qnorm = math.sqrt(qnorm2) if qnorm2 > 0 else 1.0
        dnorm = self.char_doc_norm.get(docid, 1.0)
        return dot / (qnorm * dnorm) if qnorm > 0 and dnorm > 0 else 0.0

    def rank(self, query: str, top_k: int = 1000) -> list[RetrievedDoc]:
        q_tokens = self.tokenize(query)
        q_lower = query.lower()
        scored: list[RetrievedDoc] = []

        for docid in self.term_freqs:
            bm25_score = self.bm25(q_tokens, docid)
            char_score = self.char_tfidf_cosine(q_lower, docid)
            humor = self.humor_prior.get(docid, 0.0)
            exact = 1.0 if q_lower and q_lower in self.doc_text_lower[docid] else 0.0
            score = (
                self.bm25_weight * bm25_score
                + self.char_weight * char_score
                + self.humor_weight * humor
                + self.match_boost * exact
            )
            if score > 0.0:
                scored.append(RetrievedDoc(docid=docid, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def normalize_scores(rows: list[RetrievedDoc]) -> list[RetrievedDoc]:
        if not rows:
            return rows
        max_score = max(r.score for r in rows)
        min_score = min(r.score for r in rows)
        if max_score == min_score:
            return [RetrievedDoc(docid=r.docid, score=1.0) for r in rows]
        return [
            RetrievedDoc(docid=r.docid, score=(r.score - min_score) / (max_score - min_score))
            for r in rows
        ]
