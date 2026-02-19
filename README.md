# JOKER @ CLEF 2025 — Task 1 Mini Project

This repository contains a practical retrieval pipeline for **Task 1: Humour-aware information retrieval**.

## Input files

For EN task, use:
- `joker_task1_retrieval_corpus25_EN.json`
- `joker_task1_retrieval_queries_train25_EN.json`
- `joker_task1_retrieval_queries_test25_EN.json`
- `joker_task1_retrieval_qrels_train25_EN.json`

> Make sure each file is a valid JSON array (`[...]`).

## Run with GUI (recommended)

### Start GUI

```bash
PYTHONPATH=src python -m joker_task1.gui
```

In the GUI:
1. Select corpus, queries, and optional qrels files.
2. Set `run_id`, `top-k`, and whether the run is manual.
3. Enable **Auto-tune** if you want parameter search (requires qrels).
4. Click **Run Prediction**.
5. Watch live progress bar + logs (indexing, tuning, ranking, saving).

The GUI writes:
- `prediction.json`
- optional zip with `prediction.json` at root for Codabench submission.

## Run with CLI

### Build prediction for test set

```bash
PYTHONPATH=src python -m joker_task1.cli predict \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_test25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --run-id YOURTEAM_task_1_hybrid_en \
  --manual 0 \
  --output prediction.json \
  --zip submission.zip
```

### Auto-tune on train queries

```bash
PYTHONPATH=src python -m joker_task1.cli predict \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --auto-tune \
  --run-id YOURTEAM_task_1_hybrid_en_tuned \
  --manual 0 \
  --output prediction_train_tuned.json
```

### Evaluate MAP@K

```bash
PYTHONPATH=src python -m joker_task1.cli eval \
  --predictions prediction_train_tuned.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  -k 1000
```

## Retrieval method

Hybrid score combines:
1. BM25 on word tokens
2. Character 3–5gram TF-IDF cosine
3. Humor prior from training qrels
4. Exact query substring boost

This stays dependency-free (stdlib only) and is suitable for fast experimentation.
