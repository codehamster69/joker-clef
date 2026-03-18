# Approach comparison for JOKER CLEF Task 1

This document compares the repository's upgraded hybrid pipeline against:
- the original repo baseline,
- classical lexical approaches such as Indri / RM3,
- representative CLEF JOKER 2024/2025 systems,
- and practical Hugging Face model choices.

---

## 1) Original repo baseline

### Method
- BM25 word retrieval
- character 3–5gram TF-IDF cosine
- humor prior from train qrels
- exact query substring boost

### Strengths
- extremely lightweight
- no external ML dependencies
- easy to tune and debug
- good at lexical overlap and spelling-style cues

### Weaknesses
- limited semantic matching
- no neural reranking
- humor prior is global, not query-conditioned
- weaker on indirect, paraphrastic, or semantically distant humor matches

### Verdict
A strong baseline, but not the best ceiling for CLEF-style humor-aware retrieval.

---

## 2) Indri / RM3 / classical lexical retrieval

### Method family
- query likelihood / lexical matching
- pseudo relevance feedback (RM3)
- structured query operators
- strong traditional IR indexing

### Strengths
- reliable and interpretable
- strong first-stage retrieval
- RM3 often improves recall
- good baseline for comparison

### Weaknesses
- still fundamentally lexical
- does not explicitly model humor compatibility
- struggles when the query and relevant joke/pun are semantically related but lexically different

### Verdict
Important baseline to report, but typically outperformed by modern lexical+dense hybrids.

---

## 3) CLEF 2024-style approaches

### Typical pattern
- BM25 / Anserini / RM3 retrieval
- then humor or relevance filtering using a neural model

### Strengths
- stronger than plain lexical baselines
- practical and reproducible
- simple 2-stage architecture

### Weaknesses
- success depends heavily on the quality of the filter/classifier
- retrieval stage can still miss semantically distant candidates

### Verdict
Very relevant historical baseline. Our upgraded system should outperform this family if dense retrieval and reranking are tuned well.

---

## 4) CLEF 2025 PICT-style modular hybrid

### Typical pattern
- TF-IDF/BM25 retrieval
- RM3 / expansion
- semantic branch
- cross-encoder or late-interaction reranking
- wordplay-oriented engineered signals

### Strengths
- highly competitive practical architecture
- modular and easy to ablate
- combines lexical, semantic, and humor cues

### Weaknesses
- more engineering effort
- more moving parts to tune

### Verdict
This is one of the closest published families to the architecture implemented in this repo.

---

## 5) CLEF 2025 relevance-aware classification approach

### Typical pattern
- semantic retriever
- query-document classifier for humor/relevance

### Strengths
- elegant and task-aligned
- models query-conditioned relevance directly
- practical on moderate hardware

### Weaknesses
- can underperform if the first-stage retrieval is not strong enough
- classification-only reranking may miss lexical pun cues unless combined with a lexical branch

### Verdict
A strong design inspiration. Our new pipeline keeps this idea but adds a more robust lexical+dense fusion front-end.

---

## 6) CLEF 2025 Qwen-style large-model pipeline

### Typical pattern
- large embedding model for retrieval
- large LLM for humor filtering
- explanation generation / reasoning stage

### Strengths
- highest raw performance direction in published 2025 task results
- captures nuanced humor and semantic relations

### Weaknesses
- heavy compute cost
- difficult to run locally on a laptop GPU
- slower iteration and higher operational complexity

### Verdict
Best as a reference ceiling, not as the default practical implementation for this repo.

---

## 7) Hugging Face model comparison

| Model | Role | Pros | Cons | Verdict |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Dense retriever | Strong retrieval quality, compact, practical on laptop GPU | English only | **Default dense retriever in this repo** |
| `intfloat/e5-small-v2` | Dense retriever | Safe, efficient baseline | Slightly weaker practical default than BGE-small for this use case | Good fallback |
| `BAAI/bge-m3` | Dense retriever | Powerful multilingual option | Heavier for a 3050 Ti laptop | Future upgrade |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | Reranker | Strong cost/performance | Not humor-specific | **Default reranker** |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker | Strong multilingual reranking | Heavier and more expensive | Upgrade option |
| `roberta-base` | Humor-aware scorer | Practical to fine-tune locally | Needs training data and tuning | **Default humor classifier** |
| `xlm-roberta-base` | Humor-aware scorer | Better multilingual path | Slightly heavier | Good future variant |
| Qwen embedding / LLM stack | Retrieval + filtering | Highest ceiling | Impractical for local iteration | Reference only |

---

## 8) Our upgraded repo approach

### Implemented design
- lexical baseline branch retained
- dense retrieval branch added
- reciprocal-rank candidate fusion
- cross-encoder reranking hook added
- humor-aware pair classifier training/inference added
- handcrafted overlap/humor feature fusion added
- ablation command added for structured comparison

### Why this is the best practical choice
This design is the best middle ground between:
- classical IR strength,
- modern semantic retrieval,
- humor-aware ranking,
- and laptop-scale feasibility.

### Best-fit hardware assumption
This repo's new default configuration is optimized for a laptop GPU such as an **NVIDIA RTX 3050 Ti**.

---

## 9) Summary ranking of approaches

### Best raw ceiling
1. Qwen-style large-model CLEF 2025 pipelines

### Best practical research architecture
2. Lexical + dense + reranker + humor-aware scorer (**our upgraded repo approach**)

### Best classical baseline
3. Indri / BM25 / RM3

### Best ultra-lightweight baseline
4. Original lexical-only repo baseline

---

## 10) Recommended reporting table for your experiments

Use these rows in your report:
- Original lexical baseline
- BM25 / classical lexical baseline
- BM25 + RM3 baseline
- Dense-only baseline
- Lexical + dense
- Lexical + dense + reranker
- Lexical + dense + humor scorer
- Full hybrid (current recommended system)
- CLEF 2024 reference family
- CLEF 2025 modular hybrid reference family
- CLEF 2025 Qwen-style reference ceiling
