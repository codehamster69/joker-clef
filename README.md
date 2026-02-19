# JOKER @ CLEF 2025 — Task 1 Mini Project

This repository contains a practical retrieval pipeline for **Task 1: Humour-aware information retrieval**.

## Expected data files

Place your Task 1 data files anywhere in the repo and pass paths via CLI.
Typical setup is 4 JSON files:
- English corpus (`docid`, `text`)
- English queries (`qid`, `query`)
- Portuguese corpus (`docid`, `text`)
- Portuguese queries (`qid`, `query`)

If train qrels are available, pass them with `--qrels` for better ranking.

## Method (stronger efficient baseline)

Hybrid lexical score:
1. **BM25** on tokenized text.
2. **Character n-gram TF-IDF cosine** (robust for short queries, names, morphology).
3. **Humor prior** from train qrels.
4. **Exact query-substring boost**.

Everything is dependency-free (Python stdlib only).

## Setup

```bash
python -m pip install -e .
```

## Build predictions

```bash
joker-task1 predict \
  --docs data/docs_en.json \
  --queries data/queries_en_test.json \
  --qrels data/qrels_en_train.json \
  --run-id YOURTEAM_task_1_hybrid \
  --manual 0 \
  --output prediction.json \
  --zip submission.zip
```

## Auto-tune parameters (recommended)

This does a query-level holdout split over the provided qrels and picks the best parameter set by MAP.

```bash
joker-task1 predict \
  --docs data/docs_en.json \
  --queries data/queries_en_train.json \
  --qrels data/qrels_en_train.json \
  --auto-tune \
  --run-id YOURTEAM_task_1_hybrid_tuned \
  --manual 0 \
  --output prediction.json \
  --zip submission.zip
```

## Evaluate locally

```bash
joker-task1 eval \
  --predictions prediction.json \
  --qrels data/qrels_en_train.json \
  -k 1000
```

## Output format

Each row in `prediction.json`:
- `run_id`
- `manual`
- `qid`
- `docid`
- `rank`
- `score` in [0,1]

ZIP submissions must contain `prediction.json` at the root.
