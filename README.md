# JOKER @ CLEF 2025 — Task 1 Mini Project

This repository contains a practical retrieval pipeline for **Task 1: Humour-aware information retrieval**.

It now supports two execution modes:

1. **Lexical baseline** — BM25 + char n-gram TF-IDF + humor prior + exact match boost.
2. **Hybrid neural pipeline** — lexical branch + dense retrieval + reranking + humor-aware query/document scoring.

## Input files

For EN task, use:
- `joker_task1_retrieval_corpus25_EN.json`
- `joker_task1_retrieval_queries_train25_EN.json`
- `joker_task1_retrieval_queries_test25_EN.json`
- `joker_task1_retrieval_qrels_train25_EN.json`

> Make sure each file is a valid JSON array (`[...]`).

---

## Installation

```bash
pip install -e .
```

### Runtime dependencies

The hybrid pipeline uses:
- `torch`
- `transformers`
- `sentence-transformers`
- `numpy`
- `faiss-cpu`

If you only want the original lexical baseline, the old code path still exists under the `predict` command.

---

## GUI workflow

### Start GUI

```bash
PYTHONPATH=src python -m joker_task1.gui
```

### GUI now supports

- Baseline **and** hybrid pipeline selection
- Dense index building from the GUI
- Humor model training from the GUI
- Prediction run setup (docs, queries, qrels, run_id, manual, top-k, zip)
- Auto-tuning with live progress for lexical weights
- Save/load lexical parameter JSON files
- Hybrid controls for dense model, reranker, humor model, fusion config, batch size, and device
- Evaluate predictions (MAP@K) directly in GUI
- Compare multiple reranker models from the GUI and save the MAP comparison as JSON
- Live progress logs for indexing, tuning, ranking, and training
- Live PC resource monitoring (CPU, RAM, and GPU/VRAM when `nvidia-smi` is available)

---

## CLI workflow

## 1) Original lexical baseline

### Tune lexical baseline on train

```bash
PYTHONPATH=src python -m joker_task1.cli predict \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --auto-tune \
  --run-id YOURTEAM_task_1_lexical_tuned \
  --manual 0 \
  --output prediction_train_lexical.json
```

### Evaluate lexical baseline

```bash
PYTHONPATH=src python -m joker_task1.cli eval \
  --predictions prediction_train_lexical.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  -k 1000
```

---

## 2) Build dense retrieval index

```bash
PYTHONPATH=src python -m joker_task1.cli build-dense-index \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --model-name BAAI/bge-small-en-v1.5 \
  --index-dir artifacts/dense_index \
  --device cuda \
  --batch-size 32
```

This creates:
- `embeddings.npy`
- `docids.json`
- `meta.json`
- `faiss.index` (when FAISS is available)

---

## 3) Train humor-aware query/document scorer

```bash
PYTHONPATH=src python -m joker_task1.cli train-humor \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --output-dir artifacts/humor_model \
  --model-name roberta-base \
  --device cuda \
  --epochs 3 \
  --batch-size 4
```

This trains a compact query-conditioned scorer using positive qrels and lexical hard negatives.

---

## 4) Run the full hybrid pipeline

```bash
PYTHONPATH=src python -m joker_task1.cli predict-hybrid \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_test25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --run-id YOURTEAM_task_1_hybrid_en_final \
  --manual 0 \
  --output prediction.json \
  --zip submission.zip \
  --dense-model BAAI/bge-small-en-v1.5 \
  --dense-index-dir artifacts/dense_index \
  --dense-top-k 700 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L12-v2 \
  --rerank-top-n 200 \
  --humor-model-dir artifacts/humor_model \
  --device cuda
```

### What the hybrid pipeline does

1. Baseline lexical retrieval.
2. Dense retrieval using a sentence-transformer encoder.
3. Reciprocal-rank fusion of lexical and dense candidates.
4. Cross-encoder reranking over the top candidate set.
5. Humor-aware pair scoring over the reranked candidate set.
6. Final weighted fusion of all scores plus handcrafted overlap/humor features.

---

## 5) Run ablations

```bash
PYTHONPATH=src python -m joker_task1.cli ablate \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --run-id YOURTEAM_task_1_ablation \
  --output-dir artifacts/ablations \
  --dense-model BAAI/bge-small-en-v1.5 \
  --dense-index-dir artifacts/dense_index \
  --reranker-model cross-encoder/ms-marco-MiniLM-L12-v2 \
  --humor-model-dir artifacts/humor_model \
  --device cuda
```

This writes ablation runs and `ablation_metrics.json` so you can compare:
- lexical baseline
- lexical + dense
- lexical + dense + reranker
- full hybrid

---

## 6) Compare multiple reranker models and store results

```bash
PYTHONPATH=src python -m joker_task1.cli compare-models \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --run-id YOURTEAM_task_1_model_compare \
  --output-dir artifacts/model_comparisons \
  --comparison-file artifacts/model_comparisons/model_comparison_metrics.json \
  --dense-model BAAI/bge-small-en-v1.5 \
  --dense-index-dir artifacts/dense_index \
  --models camembert-base Qwen/Qwen3-Reranker-0.6B facebook/bart-base distilbert-base-uncased \
  --rerank-top-n 200 \
  --device cuda
```

This writes:
- one prediction file per candidate model in `--output-dir`
- one summary JSON (`--comparison-file`) sorted by MAP@K

---

## Top-K meaning

`top-k` is the maximum number of retrieved documents per query.

- Higher `top-k` improves recall and usually MAP potential.
- This track allows up to 1000 documents/query, so default `1000` is recommended.

---

## Retrieval methods in this repo

### Lexical baseline

Hybrid score combines:
1. BM25 on word tokens
2. Character 3–5gram TF-IDF cosine
3. Humor prior from training qrels
4. Exact query substring boost

### Hybrid neural pipeline

The new recommended stack combines:
1. Lexical baseline branch
2. Dense retrieval (`BAAI/bge-small-en-v1.5` by default)
3. Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L12-v2`)
4. Humor-aware pair classification (`roberta-base` fine-tuning)
5. Weighted fusion + handcrafted overlap/humor features

---

## Suggested laptop configuration (RTX 3050 Ti)

Recommended starting point:
- Dense model: `BAAI/bge-small-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L12-v2`
- Humor model: `roberta-base`
- Dense top-k: `700`
- Rerank top-n: `200`

This gives a strong quality/compute tradeoff for a laptop GPU.

---

## Comparison notes

See [`APPROACH_COMPARISON.md`](APPROACH_COMPARISON.md) for a written comparison between:
- the old repo baseline
- Indri-style lexical retrieval
- CLEF 2024 and 2025 approaches
- Hugging Face model choices
- the new hybrid pipeline added here
