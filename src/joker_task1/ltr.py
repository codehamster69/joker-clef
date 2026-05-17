"""Learning-to-Rank module using XGBoost.

Provides:
- LTRRanker: XGBRanker-based pairwise L2R with leave-one-query-out CV.
- AdaptiveWeightOptimizer: Coordinate-ascent optimizer for fusion weights.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fusion import CandidateDoc

FEATURE_KEYS = [
    "lexical_score",
    "dense_score",
    "rerank_score",
    "humor_score",
    "exact_match",
    "token_overlap",
    "char_overlap",
    "doc_len_norm",
    "punct_norm",
    "exclaim_norm",
    "quote_norm",
    "repeated_words_norm",
    "query_polysemy",
    "avg_word_rarity",
    "question_in_doc",
    "doc_sentiment_contrast",
]


def _candidate_to_vec(cand: "CandidateDoc") -> list[float]:
    feats = cand.feature_scores
    return [
        cand.lexical_score,
        cand.dense_score,
        cand.rerank_score,
        cand.humor_score,
        feats.get("exact_match", 0.0),
        feats.get("token_overlap", 0.0),
        feats.get("char_overlap", 0.0),
        feats.get("doc_len_norm", 0.0),
        feats.get("punct_norm", 0.0),
        feats.get("exclaim_norm", 0.0),
        feats.get("quote_norm", 0.0),
        feats.get("repeated_words_norm", 0.0),
        feats.get("query_polysemy", 0.0),
        feats.get("avg_word_rarity", 0.0),
        feats.get("question_in_doc", 0.0),
        feats.get("doc_sentiment_contrast", 0.0),
    ]


class LTRRanker:
    """XGBoost pairwise Learning-to-Rank over candidate feature vectors."""

    def __init__(self, max_depth: int = 3, n_estimators: int = 100,
                 reg_alpha: float = 0.5, reg_lambda: float = 1.0,
                 learning_rate: float = 0.1):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self._model = None

    def _build_model(self):
        try:
            from xgboost import XGBRanker
        except ImportError as exc:
            raise ImportError("pip install xgboost") from exc
        return XGBRanker(
            objective="rank:pairwise",
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            learning_rate=self.learning_rate,
            tree_method="hist",
            device="cpu",
            eval_metric="ndcg",
            verbosity=0,
        )

    def _build_matrix(self, candidates_by_qid: dict, rel_by_qid: dict,
                      exclude_qids: set | None = None):
        X_rows, y_rows, groups = [], [], []
        for qid, cands in candidates_by_qid.items():
            if exclude_qids and qid in exclude_qids:
                continue
            rel = rel_by_qid.get(qid, set())
            vecs = [(_candidate_to_vec(c), 1 if c.docid in rel else 0)
                    for c in cands.values()]
            if not vecs:
                continue
            for vec, label in vecs:
                X_rows.append(vec)
                y_rows.append(label)
            groups.append(len(vecs))
        if not X_rows:
            return None, None, None
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), groups

    def cross_validate(self, candidates_by_qid: dict, rel_by_qid: dict) -> float:
        """Leave-one-query-out CV. Returns mean AP across held-out queries."""
        from .cli import map_at_k
        qids = list(candidates_by_qid.keys())
        if len(qids) < 2:
            return 0.0
        ap_scores: list[float] = []
        for held_qid in qids:
            X_train, y_train, groups = self._build_matrix(
                candidates_by_qid, rel_by_qid, exclude_qids={held_qid}
            )
            if X_train is None or y_train.sum() == 0:
                continue
            model = self._build_model()
            model.fit(X_train, y_train, qid=_groups_to_qid(groups))
            held_cands = candidates_by_qid.get(held_qid, {})
            if not held_cands:
                continue
            X_held = np.array([_candidate_to_vec(c) for c in held_cands.values()],
                               dtype=np.float32)
            scores = model.predict(X_held)
            docids = list(held_cands.keys())
            ranked = [docids[i] for i in np.argsort(-scores)]
            pred = {held_qid: ranked}
            ap_scores.append(map_at_k(pred, rel_by_qid))
        return float(np.mean(ap_scores)) if ap_scores else 0.0

    def fit(self, candidates_by_qid: dict, rel_by_qid: dict) -> float:
        """Train on all queries. Returns leave-one-out CV MAP before final fit."""
        cv_map = self.cross_validate(candidates_by_qid, rel_by_qid)
        X, y, groups = self._build_matrix(candidates_by_qid, rel_by_qid)
        if X is None or y.sum() == 0:
            return cv_map
        self._model = self._build_model()
        self._model.fit(X, y, qid=_groups_to_qid(groups))
        return cv_map

    def predict(self, candidates: dict) -> dict[str, float]:
        """Return {docid: score} for a single query's candidates."""
        if self._model is None:
            raise RuntimeError("Call fit() or load() first.")
        if not candidates:
            return {}
        docids = list(candidates.keys())
        X = np.array([_candidate_to_vec(c) for c in candidates.values()],
                      dtype=np.float32)
        scores = self._model.predict(X)
        return dict(zip(docids, scores.tolist()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"model": self._model, "params": self._params_dict()}, f)

    def load(self, path: str | Path) -> "LTRRanker":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        for k, v in data.get("params", {}).items():
            setattr(self, k, v)
        return self

    def _params_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.learning_rate,
        }


def _groups_to_qid(groups: list[int]) -> np.ndarray:
    """Convert group sizes to XGBoost qid array (0-indexed per doc)."""
    qid = []
    for i, g in enumerate(groups):
        qid.extend([i] * g)
    return np.array(qid, dtype=np.int32)


class AdaptiveWeightOptimizer:
    """Coordinate-ascent optimizer for the 4 main fusion weights.

    Maximizes MAP@1000 on training queries without needing XGBoost —
    useful as a fast, interpretable alternative.
    """

    WEIGHT_KEYS = ["lexical", "dense", "rerank", "humor"]

    def optimize(self, candidates_by_qid: dict, rel_by_qid: dict,
                 n_iter: int = 30, step: float = 0.1,
                 progress=None) -> dict:
        """Return a weights dict like DEFAULT_FUSION_WEIGHTS."""
        from scipy.optimize import minimize
        from .fusion import weighted_fuse
        from .cli import map_at_k

        def _score(w_vec: np.ndarray) -> float:
            weights = {
                "lexical": float(w_vec[0]),
                "dense": float(w_vec[1]),
                "rerank": float(w_vec[2]),
                "humor": float(w_vec[3]),
                "feature_weights": _default_feat_weights(),
            }
            pred_by_qid: dict[str, list[str]] = {}
            for qid, cands in candidates_by_qid.items():
                ranked = weighted_fuse(cands, weights)
                pred_by_qid[qid] = [r.docid for r in ranked]
            return -map_at_k(pred_by_qid, rel_by_qid)

        x0 = np.array([1.0, 0.8, 1.2, 1.0])
        bounds = [(0.0, 5.0)] * 4
        result = minimize(
            _score, x0,
            method="Nelder-Mead",
            options={"maxiter": n_iter * 10, "xatol": 1e-4, "fatol": 1e-4},
        )
        best_w = result.x.clip(0)
        return {
            "lexical": float(best_w[0]),
            "dense": float(best_w[1]),
            "rerank": float(best_w[2]),
            "humor": float(best_w[3]),
            "feature_weights": _default_feat_weights(),
        }


def _default_feat_weights() -> dict[str, float]:
    return {
        "exact_match": 0.1,
        "token_overlap": 0.12,
        "char_overlap": 0.1,
        "doc_len_norm": 0.04,
        "punct_norm": 0.04,
        "exclaim_norm": 0.03,
        "quote_norm": 0.03,
        "repeated_words_norm": 0.04,
        "query_polysemy": 0.05,
        "avg_word_rarity": 0.04,
        "question_in_doc": 0.03,
        "doc_sentiment_contrast": 0.03,
    }
