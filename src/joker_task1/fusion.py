from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import RetrievedDoc


@dataclass
class CandidateDoc:
    docid: str
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    humor_score: float = 0.0
    feature_scores: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0


def _normalize_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    vals = list(score_map.values())
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


def rrf_fuse(*rankings: list[RetrievedDoc], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for rows in rankings:
        for rank, row in enumerate(rows, start=1):
            fused[row.docid] = fused.get(row.docid, 0.0) + 1.0 / (k + rank)
    return fused


def build_candidates(lexical_rows: list[RetrievedDoc], dense_rows: list[RetrievedDoc]) -> dict[str, CandidateDoc]:
    candidates: dict[str, CandidateDoc] = {}
    lexical_map = _normalize_map({row.docid: row.score for row in lexical_rows})
    dense_map = _normalize_map({row.docid: row.score for row in dense_rows})
    for docid in set(lexical_map) | set(dense_map):
        candidates[docid] = CandidateDoc(
            docid=docid,
            lexical_score=lexical_map.get(docid, 0.0),
            dense_score=dense_map.get(docid, 0.0),
        )
    return candidates


def weighted_fuse(candidates: dict[str, CandidateDoc], weights: dict[str, float], top_k: int = 1000) -> list[RetrievedDoc]:
    rows: list[RetrievedDoc] = []
    feature_weights = weights.get("feature_weights", {}) if isinstance(weights.get("feature_weights"), dict) else {}
    for cand in candidates.values():
        feat_total = 0.0
        for key, value in cand.feature_scores.items():
            feat_total += feature_weights.get(key, 0.0) * value
        cand.final_score = (
            weights.get("lexical", 1.0) * cand.lexical_score
            + weights.get("dense", 0.8) * cand.dense_score
            + weights.get("rerank", 1.2) * cand.rerank_score
            + weights.get("humor", 1.0) * cand.humor_score
            + feat_total
        )
        rows.append(RetrievedDoc(docid=cand.docid, score=cand.final_score))
    rows.sort(key=lambda row: row.score, reverse=True)
    return rows[:top_k]
