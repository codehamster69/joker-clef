# GUI Instructions — JOKER Task 1 Retriever

This document walks through every panel and button in the GUI, explains what each control does, and provides a recommended step-by-step workflow from raw data to final submission.

## Launch the GUI

```bash
joker-task1-gui
# or
python -m joker_task1.gui
```

A 1220×920 scrollable window opens titled **"JOKER Task 1 Retriever"**.

---

## Panel Overview (top to bottom)

```
┌─────────────────────────────────────────────┐
│  Task files                                 │  ← Set all file paths here
├─────────────────────────────────────────────┤
│  Run controls                               │  ← Pipeline mode, device, etc.
├─────────────────────────────────────────────┤
│  Hybrid settings                            │  ← Dense model, reranker
├─────────────────────────────────────────────┤
│  XGBoost L2R / Adaptive weights             │  ← LTR model (NEW)
├─────────────────────────────────────────────┤
│  Pseudo-Relevance Feedback (PRF)            │  ← Query expansion (NEW)
├─────────────────────────────────────────────┤
│  Humor model training                       │  ← Train roberta classifier
├─────────────────────────────────────────────┤
│  Action buttons                             │  ← Click to run operations
├─────────────────────────────────────────────┤
│  Prediction file to evaluate                │  ← Evaluate existing file
├─────────────────────────────────────────────┤
│  Model comparison (reranker)                │  ← Compare multiple rerankers
├─────────────────────────────────────────────┤
│  Execution status and resource usage        │  ← Progress bar + system stats
├─────────────────────────────────────────────┤
│  Logger                                     │  ← Scrollable log output
└─────────────────────────────────────────────┘
```

---

## 1. Task Files Panel

Set all file and directory paths before running anything.

| Field | What to put here | Example |
|-------|-----------------|---------|
| **Corpus JSON** | Path to the 77k document corpus | `joker_task1_retrieval_corpus25_EN.json` |
| **Queries JSON** | Training or test queries | `joker_task1_retrieval_queries_train25_EN.json` |
| **Qrels JSON** | Relevance labels (training only) | `joker_task1_retrieval_qrels_train25_EN.json` |
| **Output prediction.json** | Where to write predictions | `prediction_train.json` |
| **Output ZIP** | Optional submission archive | `submission.zip` |
| **Load lexical params JSON** | Pre-tuned BM25 params to reuse | `tuned_params.json` (leave blank to tune fresh) |
| **Save lexical params JSON** | Where to save tuned params | `tuned_params.json` |
| **Dense index directory** | Where the FAISS index is stored | `artifacts/dense_index` |
| **Humor model directory** | Trained humor classifier location | `artifacts/humor_model` |
| **Fusion config JSON** | Custom static fusion weights | Leave blank to use defaults |
| **Comparison output dir** | Where per-model predictions go | `artifacts/model_comparisons` |
| **Comparison summary JSON** | Summary file for model comparison | `artifacts/model_comparisons/model_comparison_metrics.json` |
| **Auto report JSON** | Where run reports are auto-saved | `artifacts/gui_reports/latest_run_report.json` |

Click **Browse** next to any field to use a file/folder picker.

---

## 2. Run Controls Panel

| Control | Options | What it does |
|---------|---------|-------------|
| **Pipeline** | `baseline` / `hybrid` | Baseline = BM25 only. Hybrid = full multi-stage pipeline. |
| **Run ID** | Text string | Identifier embedded in prediction JSON (e.g., `myteam_task1_hybrid`) |
| **Device** | `cuda` / `cpu` | Use GPU (recommended) or CPU for model inference |
| **Batch size** | 1–256 | Inference batch size. 32 works well for 4 GB VRAM. Reduce if OOM. |
| **Manual run** | 0 / 1 | CLEF submission flag (0 = automatic system) |
| **Auto-tune lexical weights** | Checkbox | Grid-search BM25 k1, b, char_weight on a query holdout split |
| **Evaluate after prediction** | Checkbox | Automatically compute all 6 metrics after the run completes |
| **Auto-save run report** | Checkbox | Save a JSON report of settings + metrics to the auto-report path |
| **Top-K** | 1–1000 | Number of documents to retrieve per query (1000 = CLEF standard) |

---

## 3. Hybrid Settings Panel

Only active when **Pipeline = hybrid**.

| Control | Default | What it does |
|---------|---------|-------------|
| **Dense model** | `BAAI/bge-base-en-v1.5` | HuggingFace model name for dense retrieval. Can be any sentence-transformers model. |
| **Dense top-k** | 700 | How many docs the dense retriever returns before fusion |
| **Reranker model** | `cross-encoder/ms-marco-MiniLM-L12-v2` | HuggingFace cross-encoder model name. Leave blank to skip reranking. |
| **Rerank top-n** | 200 | How many candidates the reranker scores |

**Tip**: The dense model and reranker can be changed to any compatible HuggingFace model. The GUI downloads them automatically on first use.

---

## 4. XGBoost L2R / Adaptive Weights Panel (NEW)

Replaces static fusion weights with a trained XGBoost ranking model.

| Control | Default | What it does |
|---------|---------|-------------|
| **Use XGBoost L2R scoring** | Off | Toggle: when enabled, LTR scores replace the static weighted fusion |
| **LTR model (.pkl)** | `artifacts/ltr_model.pkl` | Path to save/load the trained XGBoost model |
| **Max depth** | 3 | XGBoost tree depth. Keep at 3 for small datasets (12 queries). |
| **N estimators** | 100 | Number of boosting trees. 100 is a good default. |
| **Train LTR Model** button | — | Runs the full pipeline on training data, trains XGBoost, saves model |

**Workflow**:
1. Set all Task files paths (corpus, training queries, training qrels)
2. Click **Train LTR Model** — this may take several minutes
3. Check the Logger for "CV MAP (leave-one-query-out)" to see how well it generalizes
4. Tick **Use XGBoost L2R scoring**
5. Click **Run Prediction** — the LTR model will be used instead of static weights

---

## 5. Pseudo-Relevance Feedback (PRF) Panel (NEW)

Expands short queries (like "colors") using vocabulary from the initial top results, improving recall.

| Control | Default | What it does |
|---------|---------|-------------|
| **Enable PRF query expansion (RM3)** | Off | Toggle: expands the query before the main retrieval pass |
| **Feedback docs (K)** | 10 | How many initial top documents to draw expansion terms from |
| **Expansion terms** | 15 | How many new terms to add to the query |

**When to use**: Always helpful for very short (1-word) queries. May hurt slightly for longer, more specific queries. Test with and without to compare MAP.

---

## 6. Humor Model Training Panel

Fine-tune a transformer as a humor relevance classifier.

| Control | Default | What it does |
|---------|---------|-------------|
| **Train model** | `roberta-base` | Base model to fine-tune (can use any HuggingFace transformer) |
| **Epochs** | 3 | Training epochs. 3–5 is usually enough. |
| **Batch** | 4 | Training batch size. Keep at 4 for 4 GB VRAM. |
| **Negatives/positive** | 3 | Hard negatives per positive example (from BM25 top-50) |
| **Learning rate** | 2e-5 | Standard BERT fine-tuning LR |
| **Max length** | 256 | Maximum token length for the (query, doc) pair |

**Workflow**: Set Corpus, Queries, and Qrels paths → click **Train Humor Model** → wait ~5–15 minutes → model saved to Humor model directory.

---

## 7. Action Buttons

| Button | What it does | Prerequisite |
|--------|-------------|-------------|
| **Run Prediction** | Execute the configured pipeline and write prediction.json | Corpus + Queries paths set |
| **Compare Models** | Benchmark multiple rerankers and write a ranked summary JSON | Corpus + Queries + Qrels set; models listed in the comparison panel |
| **Build Dense Index** | Encode all 77k documents into FAISS embeddings | Corpus path set; dense model name set |
| **Train Humor Model** | Fine-tune the humor pair classifier | Corpus + Queries + Qrels set |
| **Evaluate Existing Predictions** | Compute MAP, NDCG, MRR, Recall, P@10 on a saved prediction file | Prediction file + Qrels set |
| **Train LTR Model** | Train XGBoost L2R on training queries | Corpus + Queries + Qrels + Dense index ready |
| **Clear Log** | Wipe the logger text | — |

---

## 8. Prediction File to Evaluate Panel

Use this to evaluate any existing prediction JSON without re-running the full pipeline.

1. Click **Browse** → select a `prediction_train.json` file
2. Make sure **Qrels JSON** is set in the Task files panel
3. Click **Evaluate Existing Predictions**
4. All 6 metrics print in the Logger and the status bar

---

## 9. Model Comparison Panel

Compare multiple reranker models in one run.

- Enter model names separated by spaces in the text field
- Default: `camembert-base Qwen/Qwen3-Reranker-0.6B facebook/bart-base distilbert-base-uncased`
- Click **Compare Models**
- Results are saved to the comparison summary JSON, sorted by MAP@K (best first)

---

## 10. Status Bar

| Element | What it shows |
|---------|--------------|
| **Progress bar** | 0–100% progress of the current operation |
| **Status label** | Last logged message from the running task |
| **Resource monitor** | CPU%, RAM usage, GPU utilization, VRAM used, GPU temperature (updated every 1.5 seconds) |

Example resource display:
```
CPU: 23.4% | RAM: 68.2% (10.9/16.0 GB) | GPU: 87% | VRAM: 1842/4096 MiB | Temp: 71°C
```

---

## 11. Auto-Reports

When **Auto-save run report** is enabled, the GUI saves a structured JSON after every run:

```json
{
  "timestamp_utc": "2025-05-17T12:34:56Z",
  "event": "predict",
  "pipeline": "hybrid",
  "duration_seconds": 142.3,
  "inputs": { "docs": "...", "queries": "...", "qrels": "..." },
  "outputs": { "predictions": "...", "zip": "...", "rows_written": 219000 },
  "evaluation": {
    "MAP@1000": 0.2834,
    "NDCG@1000": 0.3120,
    "MRR": 0.4502,
    "Recall@1000": 0.6741,
    "P@10": 0.1833,
    "R-Precision": 0.2150
  },
  "settings": { "dense_model": "BAAI/bge-base-en-v1.5", "use_ltr": true, ... },
  "resource_snapshot": "CPU: 18.2% | RAM: 71.0% | GPU: 91% | ..."
}
```

---

## 12. Recommended Workflow (First Run to Submission)

Follow these steps in order for the best results:

### Step 1 — Set file paths
- Corpus JSON: `joker_task1_retrieval_corpus25_EN.json`
- Training Queries JSON: `joker_task1_retrieval_queries_train25_EN.json`
- Qrels JSON: `joker_task1_retrieval_qrels_train25_EN.json`
- Dense index directory: `artifacts/dense_index`
- Humor model directory: `artifacts/humor_model`

### Step 2 — Build Dense Index (one-time, ~10–20 min)
- Dense model: `BAAI/bge-base-en-v1.5`
- Click **Build Dense Index**
- Index saved to `artifacts/dense_index/`

### Step 3 — Train Humor Classifier (~5–15 min)
- Click **Train Humor Model** (default settings: roberta-base, 3 epochs)
- Model saved to `artifacts/humor_model/`

### Step 4 — Train LTR Model (~5–10 min)
- Ensure Dense index and Humor model are ready
- Click **Train LTR Model**
- Check Logger for CV MAP score
- Model saved to `artifacts/ltr_model.pkl`

### Step 5 — Tune Lexical Parameters (optional, ~3 min)
- Tick **Auto-tune lexical weights**
- Run a quick **baseline** prediction to tune BM25 params
- Tuned params auto-saved to `tuned_params.json`
- Load them in future runs via "Load lexical params JSON"

### Step 6 — Run Full Hybrid Prediction (Training queries, evaluate)
- Switch Queries JSON to training queries
- Enable: **Use XGBoost L2R scoring**, **Enable PRF**, **Evaluate after prediction**
- Click **Run Prediction**
- Review all 6 metrics in the Logger

### Step 7 — Switch to Test Queries and Submit
- Switch Queries JSON to: `joker_task1_retrieval_queries_test25_EN.json`
- Switch Output to: `prediction.json`
- Set Output ZIP to: `submission.zip`
- Update Run ID to your team's official run ID
- Click **Run Prediction**
- Upload `submission.zip` to the CLEF submission portal

---

## Common Errors and Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| "Corpus JSON path is invalid" | File doesn't exist at that path | Click Browse and select the correct file |
| "Auto-tune requires qrels" | Qrels path not set | Fill in the Qrels JSON field |
| CUDA out of memory | Batch size too large | Reduce Batch size to 8 or 16 |
| LTR model not found | Model not trained yet | Click **Train LTR Model** first |
| "No training examples found" | Qrels don't match the corpus | Check that docids in qrels exist in the corpus |
| Dense index not found | Index not built yet | Click **Build Dense Index** first |
