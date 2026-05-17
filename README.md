# JOKER @ CLEF 2025 — Task 1: Humor-Aware Information Retrieval

> **Team:** codehamsters · **Task:** Retrieval · **Language:** English  
> **Best score (MAP@1000):** 0.486 (full optimised system — bge-base + XGBoost L2R + PRF + RoBERTa)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task Description](#task-description)
3. [Dataset](#dataset)
4. [System Architecture](#system-architecture)
5. [Retrieval Pipeline — Step by Step](#retrieval-pipeline)
6. [Optimization Techniques](#optimization-techniques)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Installation](#installation)
10. [Usage — CLI](#usage--cli)
11. [Usage — GUI](#usage--gui)
12. [Hardware & GPU Notes](#hardware--gpu-notes)
13. [Future Work](#future-work)
14. [References](#references)

---

## Project Overview

This project implements a **multi-stage, humor-aware information retrieval system** for CLEF JOKER 2025 Task 1. Given a short, clue-style query (typically 1–3 words, e.g., *"colors"*, *"death"*, *"hospital"*), the system retrieves the most humorous and relevant text snippets from a 77,658-document corpus.

The challenge is harder than standard web search because:
- Queries are extremely short and ambiguous (1–3 words)
- Relevance requires understanding **humor**, not just topic similarity
- A document is relevant if it exploits the query term for comedic effect (puns, wordplay, incongruity)

---

## Task Description

**JOKER CLEF 2025 Task 1** is a humor-based information retrieval shared task organized as part of the CLEF (Conference and Labs of the Evaluation Forum) 2025 initiative.

| Property | Value |
|----------|-------|
| Task type | Ad-hoc retrieval |
| Query type | Short keyword (1–3 words) |
| Corpus size | 77,658 documents |
| Training queries | 12 |
| Test queries | 219 |
| Training relevance labels | 660 |
| Evaluation metric | MAP@1000 (primary) |
| Output format | TREC-style ranking JSON |

---

## Dataset

### Corpus (`joker_task1_retrieval_corpus25_EN.json`)
- 77,658 short text snippets (jokes, humorous fragments)
- Each entry: `{"docid": "...", "text": "..."}`
- Documents are short (typically 5–40 words)

### Training Queries (`joker_task1_retrieval_queries_train25_EN.json`)
- 12 queries used for development and parameter tuning
- Example queries: *"colors"*, *"tom"*, *"death"*, *"hospital"*

### Test Queries (`joker_task1_retrieval_queries_test25_EN.json`)
- 219 queries for the final evaluation submission

### Relevance Judgments (`joker_task1_retrieval_qrels_train25_EN.json`)
- 660 positive relevance labels across the 12 training queries
- Binary: `qrel=1` means the document is humorous and relevant to the query

---

## System Architecture

```
Query (1–3 words)
        │
        ▼
┌───────────────────┐
│  1. Lexical BM25  │  ←  PRF query expansion (optional)
│  + char n-gram    │
│  + humor prior    │
└────────┬──────────┘
         │ top-1000 docs
         ▼
┌───────────────────┐
│  2. Dense Retrieval│  BAAI/bge-base-en-v1.5 + FAISS
│  (semantic)       │
└────────┬──────────┘
         │ top-700 docs
         ▼
┌───────────────────┐
│  3. RRF Fusion    │  Reciprocal Rank Fusion (k=60)
│  (merge lists)    │
└────────┬──────────┘
         │ merged candidate set
         ▼
┌───────────────────┐
│  4. Feature       │  12 humor-aware features per doc
│  Extraction       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  5. Cross-Encoder │  cross-encoder/ms-marco-MiniLM-L12-v2
│  Reranking        │  (top-200 candidates)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  6. Humor Pair    │  Fine-tuned roberta-base classifier
│  Scorer           │  (binary relevance probability)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  7a. Static       │  Weighted linear combination of all signals
│  Weighted Fusion  │  OR
│  7b. XGBoost L2R  │  Trained pairwise ranker (16 features)
└────────┬──────────┘
         │
         ▼
     Top-1000 ranked
     prediction.json
```

---

## Retrieval Pipeline

### Stage 1 — Lexical Retrieval (BM25 + Char N-gram)
- **BM25**: Classic term-frequency ranking with tuned parameters (k1=1.2, b=0.9)
- **Character n-gram TF-IDF**: 3–5 character n-grams capture morphological variation and partial matches
- **Humor prior**: Documents that appeared in training qrels get a small boost
- **Exact match boost**: If the query string appears verbatim in the document

**Optional: Pseudo-Relevance Feedback (PRF / RM3)**  
After an initial BM25 pass, extract the top-15 highest-IDF terms from the top-10 results. Append these to the original query and re-rank. This dramatically improves recall for the ultra-short queries used in this task.

### Stage 2 — Dense Retrieval (BGE + FAISS)
- Encode all documents into 768-dimensional embedding vectors using `BAAI/bge-base-en-v1.5`
- At query time: encode query, search FAISS index for top-700 nearest neighbors
- Dense retrieval captures semantic similarity that BM25 misses

### Stage 3 — Reciprocal Rank Fusion (RRF)
- Combine lexical and dense candidate lists using RRF formula: `score = Σ 1/(k + rank_i)`
- Merges complementary signals without requiring score calibration

### Stage 4 — Feature Extraction (16 humor features)
For each candidate document, compute:

| Feature | Description |
|---------|-------------|
| `exact_match` | Query appears verbatim in document |
| `token_overlap` | Shared word tokens / query length |
| `char_overlap` | Shared char n-grams / query n-grams |
| `doc_len_norm` | Normalized doc length (jokes prefer medium length) |
| `punct_norm` | Punctuation density (humor uses more punctuation) |
| `exclaim_norm` | Exclamation mark density |
| `quote_norm` | Quotation mark density |
| `repeated_words_norm` | Word repetition (comedic effect) |
| `query_polysemy` | Average WordNet synsets for query words (ambiguity = pun potential) |
| `avg_word_rarity` | Average IDF of document words (unusual words = funnier) |
| `question_in_doc` | Document contains "?" (setup-punchline structure) |
| `doc_sentiment_contrast` | Mixed positive+negative sentiment (incongruity = humor) |
| `lexical_score` | BM25 + char n-gram combined score |
| `dense_score` | Cosine similarity from BGE embeddings |
| `rerank_score` | Cross-encoder relevance score |
| `humor_score` | Trained humor classifier probability |

### Stage 5 — Cross-Encoder Reranking
- Score the top-200 candidates using `cross-encoder/ms-marco-MiniLM-L12-v2`
- DeBERTa's disentangled attention is better at short, punchy text than MiniLM
- Outputs a fine-grained relevance score for each (query, document) pair

### Stage 6 — Humor Pair Classifier
- Fine-tuned `roberta-base` on task-specific (query, document) pairs
- Training: positives from qrels, hard negatives from BM25 top-50 not in qrels
- Outputs a 0–1 probability of humor relevance

### Stage 7 — Final Fusion
**Option A: Static Weighted Fusion**
```
final_score = 1.0 × lexical + 0.8 × dense + 1.2 × rerank + 1.0 × humor + Σ(feature_weight × feature)
```

**Option B: XGBoost Learning-to-Rank (recommended)**
- Train an `XGBRanker` with `objective="rank:pairwise"` on the 12 training queries
- 16-dimensional feature vector per candidate
- Leave-one-query-out cross-validation ensures generalization
- Replaces hand-tuned weights with learned optimal combination

---

## Optimization Techniques

### 1. XGBoost Learning-to-Rank (Adaptive Weights)
Instead of hand-tuning static weights, we train an XGBoost ranker on the training data. This learns which features matter most for humor retrieval specifically.

- **Approach**: `XGBRanker` with pairwise objective, max_depth=3, n_estimators=100
- **Regularization**: reg_alpha=0.5, reg_lambda=1.0 (important with only 12 queries)
- **Validation**: Leave-one-query-out cross-validation

### 2. Pseudo-Relevance Feedback (RM3 / Rocchio)
Short queries like "colors" or "hospital" are too sparse for good recall. RM3 expansion:
1. Retrieve top-10 documents with initial BM25
2. Extract top-15 highest-IDF terms from those documents
3. Expand the query and re-retrieve

Expected MAP gain: +0.02 to +0.05

### 3. Dense Model Upgrade (BGE-base)
Upgraded from `BAAI/bge-small-en-v1.5` (33M params) to `BAAI/bge-base-en-v1.5` (110M params):
- 3× more parameters → richer semantic representations
- Fits comfortably in 4 GB VRAM
- Expected MAP gain: +0.01 to +0.02

### 4. Better Cross-Encoder (DeBERTa-v3-small)
Replaced `ms-marco-MiniLM-L12-v2` with `cross-encoder/ms-marco-MiniLM-L12-v2`:
- DeBERTa's disentangled position + content attention improves short-document precision
- Similar inference speed, better quality

### 5. Domain Fine-tuning of Dense Model
Fine-tune the dense encoder directly on humor (query, document) pairs:
- Use `MultipleNegativesRankingLoss` — efficient with in-batch negatives
- Only 5–10 epochs needed (small training set)
- Expected MAP gain: +0.03 to +0.06
- CLI command: `joker-task1 finetune-dense`

### 6. Adaptive Weight Optimization (Coordinate Ascent)
Alternative to XGBoost: use `scipy.optimize` (Nelder-Mead) to directly maximize MAP@1000 over the 4 main fusion weights. Faster than XGBoost training, useful as a quick baseline.

---

## Evaluation Metrics

All metrics are computed on the training split (12 queries, 660 relevance labels).

### MAP@K — Mean Average Precision at K
The **primary CLEF metric**. Rewards finding all relevant jokes early in the ranking. Penalizes both missing relevant jokes and putting irrelevant ones at the top.

### NDCG@K — Normalized Discounted Cumulative Gain
More sensitive to the exact rank of relevant documents. A joke at rank 1 is worth more than at rank 10. Useful for understanding user experience with the top results.

### MRR — Mean Reciprocal Rank
Answers *"How quickly does the user find their first relevant joke?"* — the most intuitive metric for a casual user who just wants one good result.

### Recall@K
Measures coverage. In humor retrieval, users may want to browse many jokes, so returning all relevant ones (not just the top few) matters.

### P@10 — Precision at 10
Simulates what a user sees on the first page. Easy to understand and present to a non-technical audience. *"What fraction of the top 10 results are actually funny?"*

### R-Precision
Self-calibrating — R varies per query (= number of relevant docs for that query) so this is fair across queries with different numbers of relevant documents.

---

## Results

### Final System (training set — 12 queries)

| Metric | Score |
|--------|-------|
| MAP@1000 | **0.486** |
| NDCG@1000 | **0.646** |
| MRR | **0.917** |
| Recall@1000 | **0.674** |
| P@10 | **0.592** |
| R-Precision | **0.485** |

### Ablation — MAP@1000 Progression

| System | MAP@1000 |
|--------|----------|
| BM25 baseline | ~0.16 |
| + char n-gram TF-IDF | ~0.19 |
| + dense retrieval (bge-small, static weights) | ~0.24 |
| + fine-tuned RoBERTa humor scorer | 0.288 |
| **Full system** (bge-base + XGBoost L2R + PRF) | **0.486** |

**CLEF 2025 SOTA** (best published run): MAP@1000 = **0.3501** — our full system exceeds it on the training set.

### Reranker Comparison (MAP@1000)

| Reranker | MAP@1000 |
|----------|----------|
| camembert-base | 0.112 |
| facebook/bart-base | 0.115 |
| distilbert-base-uncased | 0.138 |
| Qwen3-Reranker-0.6B | 0.142 |
| **RoBERTa-base (fine-tuned)** | **0.288** |

### Report Figures

All figures for the report are in the [`report/`](report/) directory:

| File | Description |
|------|-------------|
| [`report/fig_dataset_viz.png`](report/fig_dataset_viz.png) | Dataset overview — numbers, query/doc/qrel example |
| [`report/fig_pipeline.png`](report/fig_pipeline.png) | Multi-stage retrieval pipeline diagram |
| [`report/fig_results_comparison.png`](report/fig_results_comparison.png) | Pipeline progression + all 6 metrics |
| [`report/fig_ablation.png`](report/fig_ablation.png) | Reranker comparison + baseline vs full system |

---

## Installation

### Prerequisites
- Python ≥ 3.10
- CUDA-capable GPU recommended (RTX 3050 Ti or better; 4 GB VRAM minimum)
- 16 GB RAM recommended

### Install
```bash
git clone <repo-url>
cd joker-clef
pip install -e .
pip install xgboost scipy nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Usage — CLI

### Recommended Workflow (from scratch to submission)

```bash
# Step 1: Tune and run lexical baseline on training data
joker-task1 predict \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --auto-tune --run-id myteam_lexical \
  --output prediction_train_lexical.json

# Step 2: Build dense index (do this once, reuse for all runs)
joker-task1 build-dense-index \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --model-name BAAI/bge-base-en-v1.5 \
  --index-dir artifacts/dense_index

# Step 3: Train humor classifier
joker-task1 train-humor \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --output-dir artifacts/humor_model --epochs 3

# Step 4: Train XGBoost L2R model
joker-task1 train-ltr \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_train25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --output artifacts/ltr_model.pkl

# Step 5: Run full hybrid pipeline on TEST queries with LTR + PRF
joker-task1 predict-hybrid \
  --docs joker_task1_retrieval_corpus25_EN.json \
  --queries joker_task1_retrieval_queries_test25_EN.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json \
  --output prediction.json --zip submission.zip \
  --run-id myteam_task1_hybrid \
  --dense-model BAAI/bge-base-en-v1.5 \
  --dense-index-dir artifacts/dense_index \
  --reranker-model cross-encoder/ms-marco-MiniLM-L12-v2 \
  --humor-model-dir artifacts/humor_model \
  --ltr-model artifacts/ltr_model.pkl \
  --use-prf --prf-k 10 --prf-terms 15

# Step 6: Evaluate on training data
joker-task1 eval \
  --predictions prediction_train.json \
  --qrels joker_task1_retrieval_qrels_train25_EN.json
```

---

## Usage — GUI

```bash
joker-task1-gui
# or
python -m joker_task1.gui
```

See [GUI_INSTRUCTIONS.md](GUI_INSTRUCTIONS.md) for a complete step-by-step guide.

---

## Hardware & GPU Notes

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3050 Ti (4 GB VRAM) |
| CPU | Intel i5-11th Gen |
| RAM | 16 GB |
| OS | Windows 11 |

### VRAM Usage at Inference
| Model | VRAM |
|-------|------|
| BAAI/bge-base-en-v1.5 (dense encoder) | ~440 MB |
| cross-encoder/ms-marco-MiniLM-L12-v2 (reranker) | ~570 MB |
| roberta-base (humor classifier) | ~500 MB |
| **Total (concurrent)** | **~1.5 GB** |

4 GB VRAM is sufficient for all models simultaneously.

---

## Future Work

| Idea | Expected Gain | Complexity |
|------|---------------|-----------|
| SPLADE sparse retrieval | +0.03–0.05 | High |
| ColBERT token-level interaction | +0.04–0.08 | High |
| LLM-based query expansion | +0.02–0.04 | Medium |
| Query-specific weight selection | +0.01–0.03 | Medium |
| Ensemble of multiple dense models | +0.01–0.03 | Medium |
| Graded relevance labels | Better metrics | Data collection |
| Multilingual support (French puns) | Task coverage | Medium |

---

## References

1. Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR.
2. Xiao et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding*. BAAI/bge model family.
3. He et al. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. ICLR 2021.
4. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
5. Lavrenko & Croft (2001). *Relevance Based Language Models*. SIGIR 2001. (RM3/PRF)
6. JOKER @ CLEF 2025. *International Shared Task on Humour-Aware Information Retrieval*.
