from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from random import Random
from typing import Callable

from .data import docs_by_id, load_json, save_json, to_qrel_map, zip_single_file
from .features import humor_features
from .fusion import build_candidates, rrf_fuse, weighted_fuse
from .retriever import HybridTask1Retriever

ProgressFn = Callable[[str, float], None]
DEFAULT_FUSION_WEIGHTS = {
    "lexical": 1.0,
    "dense": 0.8,
    "rerank": 1.2,
    "humor": 1.0,
    "feature_weights": {
        "exact_match": 0.1,
        "token_overlap": 0.12,
        "char_overlap": 0.1,
        "doc_len_norm": 0.04,
        "punct_norm": 0.04,
        "exclaim_norm": 0.03,
        "quote_norm": 0.03,
        "repeated_words_norm": 0.04,
    },
}


def map_at_k(pred_by_qid: dict[str, list[str]], rel_by_qid: dict[str, set[str]], k: int = 1000) -> float:
    aps: list[float] = []
    for qid, rel_set in rel_by_qid.items():
        preds = pred_by_qid.get(qid, [])[:k]
        if not rel_set:
            continue
        hit_count = 0
        precision_sum = 0.0
        for i, docid in enumerate(preds, start=1):
            if docid in rel_set:
                hit_count += 1
                precision_sum += hit_count / i
        aps.append(precision_sum / len(rel_set))
    return sum(aps) / len(aps) if aps else 0.0


def split_qrels_by_query(qrels: list[dict], valid_ratio: float = 0.2, seed: int = 13) -> tuple[list[dict], list[dict]]:
    qids = sorted({str(r["qid"]) for r in qrels})
    rnd = Random(seed)
    rnd.shuffle(qids)
    cut = max(1, int(len(qids) * valid_ratio))
    valid_qids = set(qids[:cut])
    train = [r for r in qrels if str(r["qid"]) not in valid_qids]
    valid = [r for r in qrels if str(r["qid"]) in valid_qids]
    return train, valid


def predictions_from_rankings(run_id: str, manual: int, query_rows: list[dict], rankings: dict[str, list]) -> list[dict]:
    out: list[dict] = []
    for q in query_rows:
        qid = str(q["qid"])
        ranked = rankings.get(qid, [])
        for rank, r in enumerate(HybridTask1Retriever.normalize_scores(ranked), start=1):
            out.append(
                {
                    "run_id": run_id,
                    "manual": int(manual),
                    "qid": qid,
                    "docid": r.docid,
                    "rank": rank,
                    "score": round(r.score, 6),
                }
            )
    return out


def build_predictions(
    docs_path: str,
    queries_path: str,
    output_path: str,
    run_id: str,
    manual: int,
    qrels_path: str | None = None,
    top_k: int = 1000,
    params: dict | None = None,
    progress: ProgressFn | None = None,
) -> list[dict]:
    if progress:
        progress("Loading input files...", 0.02)
    docs = load_json(docs_path)
    queries = load_json(queries_path)
    qrels = load_json(qrels_path) if qrels_path else None

    retriever = HybridTask1Retriever(**(params or {}))
    retriever.fit(docs=docs, qrels=qrels, progress=progress)

    rankings = {}
    total_queries = max(1, len(queries))
    for idx, q in enumerate(queries, start=1):
        qid = str(q["qid"])
        rankings[qid] = retriever.rank(str(q["query"]), top_k=top_k)
        if progress and (idx % 5 == 0 or idx == total_queries):
            progress(f"Ranking queries: {idx}/{total_queries}", 0.6 + 0.3 * (idx / total_queries))

    rows = predictions_from_rankings(run_id, manual, queries, rankings)
    save_json(rows, output_path)
    if progress:
        progress(f"Saved predictions to {output_path}", 0.96)
    return rows


def load_fusion_config(path: str | None) -> dict:
    if not path:
        return json.loads(json.dumps(DEFAULT_FUSION_WEIGHTS))
    data = load_json(path)
    merged = json.loads(json.dumps(DEFAULT_FUSION_WEIGHTS))
    for key, value in data.items():
        if key == "feature_weights" and isinstance(value, dict):
            merged.setdefault("feature_weights", {}).update(value)
        else:
            merged[key] = value
    return merged


def build_hybrid_predictions(
    docs_path: str,
    queries_path: str,
    output_path: str,
    run_id: str,
    manual: int,
    qrels_path: str | None = None,
    top_k: int = 1000,
    lexical_params: dict | None = None,
    dense_model: str = "BAAI/bge-small-en-v1.5",
    dense_index_dir: str = "artifacts/dense_index",
    dense_top_k: int = 700,
    reranker_model: str | None = None,
    rerank_top_n: int = 200,
    humor_model_dir: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    fusion_config_path: str | None = None,
    progress: ProgressFn | None = None,
) -> list[dict]:
    from .dense import DenseRetriever

    docs = load_json(docs_path)
    queries = load_json(queries_path)
    qrels = load_json(qrels_path) if qrels_path else None
    doc_map = docs_by_id(docs)

    if progress:
        progress("Fitting lexical retriever...", 0.02)
    lexical = HybridTask1Retriever(**(lexical_params or {}))
    lexical.fit(docs=docs, qrels=qrels, progress=progress)

    if progress:
        progress(f"Loading dense retriever ({dense_model})...", 0.12)
    dense = DenseRetriever(model_name=dense_model, index_dir=dense_index_dir, device=device, batch_size=batch_size)
    dense.ensure_ready(docs=docs, progress=progress)

    reranker = None
    if reranker_model:
        if progress:
            progress(f"Loading reranker ({reranker_model})...", 0.18)
        from .rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name=reranker_model, device=device, batch_size=max(4, batch_size // 2))

    humor_scorer = None
    if humor_model_dir:
        if progress:
            progress(f"Loading humor classifier ({humor_model_dir})...", 0.22)
        from .humor_classifier import HumorPairScorer

        humor_scorer = HumorPairScorer(model_dir=humor_model_dir, device=device)

    fusion_weights = load_fusion_config(fusion_config_path)
    rankings: dict[str, list] = {}
    total_queries = max(1, len(queries))
    for idx, query_row in enumerate(queries, start=1):
        qid = str(query_row["qid"])
        query_text = str(query_row["query"])
        lexical_rows = lexical.rank(query_text, top_k=min(top_k, dense_top_k))
        dense_rows = dense.rank(query_text, top_k=min(top_k, dense_top_k))
        fused_seed = rrf_fuse(lexical_rows, dense_rows)
        candidates = build_candidates(lexical_rows, dense_rows)
        ranked_seed = sorted(fused_seed.items(), key=lambda item: item[1], reverse=True)
        candidate_ids = [docid for docid, _ in ranked_seed[: max(rerank_top_n, 100)]]

        for docid in candidate_ids:
            if docid not in candidates:
                continue
            doc_text = str(doc_map[docid]["text"])
            candidates[docid].feature_scores = humor_features(query_text, doc_text)

        rerank_ids = candidate_ids[:rerank_top_n]
        rerank_docs = [(docid, str(doc_map[docid]["text"])) for docid in rerank_ids]
        if reranker and rerank_docs:
            reranked = reranker.rerank(query_text, rerank_docs)
            rerank_map = {row.docid: row.score for row in HybridTask1Retriever.normalize_scores(reranked)}
            for docid, score in rerank_map.items():
                if docid in candidates:
                    candidates[docid].rerank_score = score

        if humor_scorer and rerank_docs:
            humor_scores = humor_scorer.score_pairs(query_text, [text for _, text in rerank_docs], batch_size=max(4, batch_size // 2))
            for (docid, _), score in zip(rerank_docs, humor_scores):
                if docid in candidates:
                    candidates[docid].humor_score = score

        rankings[qid] = weighted_fuse(candidates, fusion_weights, top_k=top_k)
        if progress and (idx % 5 == 0 or idx == total_queries):
            progress(f"Hybrid ranking queries: {idx}/{total_queries}", 0.25 + 0.7 * (idx / total_queries))

    rows = predictions_from_rankings(run_id, manual, queries, rankings)
    save_json(rows, output_path)
    if progress:
        progress(f"Saved hybrid predictions to {output_path}", 0.98)
    return rows


def tune_params(
    docs: list[dict], queries: list[dict], qrels: list[dict], top_k: int = 1000, progress: ProgressFn | None = None
) -> tuple[dict, float]:
    train_qrels, valid_qrels = split_qrels_by_query(qrels)

    qtext = {str(q["qid"]): str(q["query"]) for q in queries}
    valid_qids = sorted({str(r["qid"]) for r in valid_qrels})
    valid_rel = to_qrel_map(valid_qrels)

    grid = {
        "k1": [1.2, 1.5, 1.8],
        "b": [0.6, 0.75, 0.9],
        "char_weight": [0.2, 0.35, 0.5],
        "humor_weight": [0.1, 0.2, 0.4],
        "match_boost": [0.05, 0.1, 0.2],
    }

    best_params: dict = {}
    best_map = -1.0

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    total = max(1, len(combos))

    for i, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        retriever = HybridTask1Retriever(**params)
        retriever.fit(docs=docs, qrels=train_qrels)

        pred_by_qid: dict[str, list[str]] = {}
        for qid in valid_qids:
            query = qtext.get(qid, "")
            pred_by_qid[qid] = [x.docid for x in retriever.rank(query, top_k=top_k)]

        score = map_at_k(pred_by_qid, valid_rel, k=top_k)
        if score > best_map:
            best_map = score
            best_params = params

        if progress and (i % 10 == 0 or i == total):
            progress(f"Auto-tune grid search: {i}/{total}", 0.05 + 0.45 * (i / total))

    return best_params, best_map


def evaluate_predictions_file(predictions_path: str, qrels_path: str, k: int) -> float:
    predictions = load_json(predictions_path)
    qrels = load_json(qrels_path)
    rel_by_qid = to_qrel_map(qrels)
    pred_by_qid: dict[str, list[str]] = {}
    for row in predictions:
        pred_by_qid.setdefault(str(row["qid"]), []).append(str(row["docid"]))
    return map_at_k(pred_by_qid, rel_by_qid, k=k)


def cmd_predict(args: argparse.Namespace) -> None:
    params = None
    if args.auto_tune:
        if not args.qrels:
            raise ValueError("--auto-tune requires --qrels")
        docs = load_json(args.docs)
        queries = load_json(args.queries)
        qrels = load_json(args.qrels)
        params, holdout_map = tune_params(docs, queries, qrels, top_k=args.top_k)
        print(f"Selected params: {params}")
        print(f"Holdout MAP@{args.top_k}: {holdout_map:.6f}")

    rows = build_predictions(
        docs_path=args.docs,
        queries_path=args.queries,
        output_path=args.output,
        run_id=args.run_id,
        manual=args.manual,
        qrels_path=args.qrels,
        top_k=args.top_k,
        params=params,
    )

    if args.zip:
        zip_single_file(args.output, args.zip, arcname="prediction.json")

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.zip:
        print(f"Created submission archive: {args.zip}")


def cmd_predict_hybrid(args: argparse.Namespace) -> None:
    rows = build_hybrid_predictions(
        docs_path=args.docs,
        queries_path=args.queries,
        output_path=args.output,
        run_id=args.run_id,
        manual=args.manual,
        qrels_path=args.qrels,
        top_k=args.top_k,
        dense_model=args.dense_model,
        dense_index_dir=args.dense_index_dir,
        dense_top_k=args.dense_top_k,
        reranker_model=args.reranker_model,
        rerank_top_n=args.rerank_top_n,
        humor_model_dir=args.humor_model_dir,
        device=args.device,
        batch_size=args.batch_size,
        fusion_config_path=args.fusion_config,
    )
    if args.zip:
        zip_single_file(args.output, args.zip, arcname="prediction.json")
    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.zip:
        print(f"Created submission archive: {args.zip}")


def cmd_build_dense_index(args: argparse.Namespace) -> None:
    from .dense import DenseRetriever

    docs = load_json(args.docs)
    retriever = DenseRetriever(model_name=args.model_name, index_dir=args.index_dir, device=args.device, batch_size=args.batch_size)
    retriever.build(docs)
    print(f"Dense index written to {args.index_dir}")


def cmd_train_humor(args: argparse.Namespace) -> None:
    from .humor_classifier import train_humor_pair_classifier

    docs = load_json(args.docs)
    queries = load_json(args.queries)
    qrels = load_json(args.qrels)
    metrics = train_humor_pair_classifier(
        docs=docs,
        queries=queries,
        qrels=qrels,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(json.dumps(metrics, indent=2))


def cmd_ablate(args: argparse.Namespace) -> None:
    if not args.qrels:
        raise ValueError("--qrels is required for ablation")
    run_specs = [
        {
            "name": "lexical",
            "kwargs": {
                "docs_path": args.docs,
                "queries_path": args.queries,
                "output_path": str(Path(args.output_dir) / "lexical.json"),
                "run_id": f"{args.run_id}_lexical",
                "manual": args.manual,
                "qrels_path": args.qrels,
                "top_k": args.top_k,
            },
            "hybrid": False,
        },
        {
            "name": "lexical_dense",
            "kwargs": {
                "docs_path": args.docs,
                "queries_path": args.queries,
                "output_path": str(Path(args.output_dir) / "lexical_dense.json"),
                "run_id": f"{args.run_id}_lexical_dense",
                "manual": args.manual,
                "qrels_path": args.qrels,
                "top_k": args.top_k,
                "dense_model": args.dense_model,
                "dense_index_dir": args.dense_index_dir,
                "dense_top_k": args.dense_top_k,
                "device": args.device,
                "batch_size": args.batch_size,
                "fusion_config_path": args.fusion_config,
            },
            "hybrid": True,
        },
    ]
    if args.reranker_model:
        run_specs.append(
            {
                "name": "lexical_dense_rerank",
                "kwargs": {
                    "docs_path": args.docs,
                    "queries_path": args.queries,
                    "output_path": str(Path(args.output_dir) / "lexical_dense_rerank.json"),
                    "run_id": f"{args.run_id}_lexical_dense_rerank",
                    "manual": args.manual,
                    "qrels_path": args.qrels,
                    "top_k": args.top_k,
                    "dense_model": args.dense_model,
                    "dense_index_dir": args.dense_index_dir,
                    "dense_top_k": args.dense_top_k,
                    "reranker_model": args.reranker_model,
                    "rerank_top_n": args.rerank_top_n,
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "fusion_config_path": args.fusion_config,
                },
                "hybrid": True,
            }
        )
    if args.humor_model_dir:
        run_specs.append(
            {
                "name": "full_hybrid",
                "kwargs": {
                    "docs_path": args.docs,
                    "queries_path": args.queries,
                    "output_path": str(Path(args.output_dir) / "full_hybrid.json"),
                    "run_id": f"{args.run_id}_full_hybrid",
                    "manual": args.manual,
                    "qrels_path": args.qrels,
                    "top_k": args.top_k,
                    "dense_model": args.dense_model,
                    "dense_index_dir": args.dense_index_dir,
                    "dense_top_k": args.dense_top_k,
                    "reranker_model": args.reranker_model,
                    "rerank_top_n": args.rerank_top_n,
                    "humor_model_dir": args.humor_model_dir,
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "fusion_config_path": args.fusion_config,
                },
                "hybrid": True,
            }
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics: list[dict] = []
    for spec in run_specs:
        if spec["hybrid"]:
            build_hybrid_predictions(**spec["kwargs"])
        else:
            build_predictions(**spec["kwargs"])
        score = evaluate_predictions_file(spec["kwargs"]["output_path"], args.qrels, args.top_k)
        metrics.append({"name": spec["name"], "map_at_k": score, "output": spec["kwargs"]["output_path"]})
        print(f"{spec['name']}: MAP@{args.top_k}={score:.6f}")
    save_json(metrics, Path(args.output_dir) / "ablation_metrics.json")


def cmd_eval(args: argparse.Namespace) -> None:
    score = evaluate_predictions_file(args.predictions, args.qrels, args.k)
    print(f"MAP@{args.k}: {score:.6f}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JOKER CLEF 2025 Task 1 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="Build lexical baseline prediction.json and optional zip")
    pp.add_argument("--docs", required=True, help="Path to corpus JSON")
    pp.add_argument("--queries", required=True, help="Path to queries JSON")
    pp.add_argument("--qrels", help="Optional train qrels to estimate humor prior")
    pp.add_argument("--auto-tune", action="store_true", help="Grid-search parameters on query holdout split")
    pp.add_argument("--output", default="prediction.json")
    pp.add_argument("--zip", dest="zip", help="Optional output zip path")
    pp.add_argument("--run-id", required=True)
    pp.add_argument("--manual", type=int, choices=[0, 1], default=0)
    pp.add_argument("--top-k", type=int, default=1000)
    pp.set_defaults(func=cmd_predict)

    pd = sub.add_parser("build-dense-index", help="Build and store dense embeddings/index")
    pd.add_argument("--docs", required=True)
    pd.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    pd.add_argument("--index-dir", default="artifacts/dense_index")
    pd.add_argument("--device", default=None)
    pd.add_argument("--batch-size", type=int, default=32)
    pd.set_defaults(func=cmd_build_dense_index)

    ph = sub.add_parser("predict-hybrid", help="Run lexical+dense+rerank+humor hybrid prediction pipeline")
    ph.add_argument("--docs", required=True)
    ph.add_argument("--queries", required=True)
    ph.add_argument("--qrels")
    ph.add_argument("--output", default="prediction_hybrid.json")
    ph.add_argument("--zip", dest="zip")
    ph.add_argument("--run-id", required=True)
    ph.add_argument("--manual", type=int, choices=[0, 1], default=0)
    ph.add_argument("--top-k", type=int, default=1000)
    ph.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    ph.add_argument("--dense-index-dir", default="artifacts/dense_index")
    ph.add_argument("--dense-top-k", type=int, default=700)
    ph.add_argument("--reranker-model")
    ph.add_argument("--rerank-top-n", type=int, default=200)
    ph.add_argument("--humor-model-dir")
    ph.add_argument("--device", default=None)
    ph.add_argument("--batch-size", type=int, default=32)
    ph.add_argument("--fusion-config")
    ph.set_defaults(func=cmd_predict_hybrid)

    pt = sub.add_parser("train-humor", help="Train a query-conditioned humor pair classifier")
    pt.add_argument("--docs", required=True)
    pt.add_argument("--queries", required=True)
    pt.add_argument("--qrels", required=True)
    pt.add_argument("--output-dir", default="artifacts/humor_model")
    pt.add_argument("--model-name", default="roberta-base")
    pt.add_argument("--device", default=None)
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--learning-rate", type=float, default=2e-5)
    pt.add_argument("--max-length", type=int, default=256)
    pt.add_argument("--negatives-per-positive", type=int, default=3)
    pt.set_defaults(func=cmd_train_humor)

    pa = sub.add_parser("ablate", help="Run baseline and hybrid ablations on qrels")
    pa.add_argument("--docs", required=True)
    pa.add_argument("--queries", required=True)
    pa.add_argument("--qrels", required=True)
    pa.add_argument("--output-dir", default="artifacts/ablations")
    pa.add_argument("--run-id", required=True)
    pa.add_argument("--manual", type=int, choices=[0, 1], default=0)
    pa.add_argument("--top-k", type=int, default=1000)
    pa.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    pa.add_argument("--dense-index-dir", default="artifacts/dense_index")
    pa.add_argument("--dense-top-k", type=int, default=700)
    pa.add_argument("--reranker-model")
    pa.add_argument("--rerank-top-n", type=int, default=200)
    pa.add_argument("--humor-model-dir")
    pa.add_argument("--device", default=None)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--fusion-config")
    pa.set_defaults(func=cmd_ablate)

    pe = sub.add_parser("eval", help="Evaluate predictions against qrels (MAP@K)")
    pe.add_argument("--predictions", required=True)
    pe.add_argument("--qrels", required=True)
    pe.add_argument("-k", type=int, default=1000)
    pe.set_defaults(func=cmd_eval)

    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
