from __future__ import annotations

import argparse
import itertools
from random import Random
from typing import Callable

from .data import load_json, save_json, to_qrel_map, zip_single_file
from .retriever import HybridTask1Retriever

ProgressFn = Callable[[str, float], None]


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


def cmd_eval(args: argparse.Namespace) -> None:
    predictions = load_json(args.predictions)
    qrels = load_json(args.qrels)

    rel_by_qid = to_qrel_map(qrels)
    pred_by_qid: dict[str, list[str]] = {}
    for row in predictions:
        pred_by_qid.setdefault(str(row["qid"]), []).append(str(row["docid"]))

    score = map_at_k(pred_by_qid, rel_by_qid, k=args.k)
    print(f"MAP@{args.k}: {score:.6f}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JOKER CLEF 2025 Task 1 pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("predict", help="Build prediction.json and optional zip")
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
