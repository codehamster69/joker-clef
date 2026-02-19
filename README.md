# JOKER @ CLEF 2025 — Task 1 Mini Project

This repository contains a practical retrieval pipeline for **Task 1: Humour-aware information retrieval**.

## Input files

For EN task, use:
- `joker_task1_retrieval_corpus25_EN.json`
- `joker_task1_retrieval_queries_train25_EN.json`
- `joker_task1_retrieval_queries_test25_EN.json`
- `joker_task1_retrieval_qrels_train25_EN.json`

> Make sure each file is a valid JSON array (`[...]`).

---

## GUI workflow (recommended)

### Start GUI

```bash
PYTHONPATH=src python -m joker_task1.gui
```

### What GUI now supports

- Prediction run setup (docs, queries, qrels, run_id, manual, top-k, zip)
- Auto-tuning with live progress
- Save tuned parameters to JSON (e.g., `tuned_params.json`)
- Load previously tuned parameters JSON and reuse for prediction
- Evaluate predictions (MAP@K) directly in GUI
- Detailed logs of each stage (indexing, tuning, ranking, saving, evaluation)

### How to use tuned setup for test prediction (GUI)

1. Run auto-tune on train queries:
   - Queries: `joker_task1_retrieval_queries_train25_EN.json`
   - Qrels: `joker_task1_retrieval_qrels_train25_EN.json`
   - Enable **Auto-tune**
   - Set **Save tuned params JSON** to `tuned_params.json`
2. Generate final test prediction with tuned params:
   - Queries: `joker_task1_retrieval_queries_test25_EN.json`
   - Disable **Auto-tune**
   - Set **Load tuned params JSON** to `tuned_params.json`
   - Run prediction and create submission zip.

### Evaluate in GUI

- Use **Evaluate after prediction** (same run) if qrels are available.
- Or click **Evaluate Existing Predictions** and choose any `prediction.json` + qrels.

---

## CLI workflow

### 1) Tune on train and save best params (printed in log)

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

### 2) Evaluate train predictions

```bash
PYTHONPATH=src python -m joker_task1.cli eval \
  --predictions prediction_train_tuned.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  -k 1000
```

### 3) Build final test submission

```bash
PYTHONPATH=src python -m joker_task1.cli predict \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_test25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --run-id YOURTEAM_task_1_hybrid_en_final \
  --manual 0 \
  --output prediction.json \
  --zip submission.zip
```

---

## Top-K meaning

`top-k` is the maximum number of retrieved documents per query.

- Higher `top-k` improves recall and usually MAP potential.
- This track allows up to 1000 documents/query, so default `1000` is recommended.

---

## Retrieval method

Hybrid score combines:
1. BM25 on word tokens
2. Character 3–5gram TF-IDF cosine
3. Humor prior from training qrels
4. Exact query substring boost

This stays dependency-free (stdlib only) and is suitable for fast experimentation.
