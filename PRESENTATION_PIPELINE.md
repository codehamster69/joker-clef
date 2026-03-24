# Presentation: Humour-Aware Retrieval Pipeline for CLEF JOKER Task 1

**Author workflow summary**
- Task: rank humorous/relevant documents for short clue-like queries.
- Dataset scale in this repo: ~77K docs, 12 train queries, 219 test queries.
- Main metric: **MAP@1000**.

---

## Slide 1 — Problem statement (raw concept)

Given a query \(q\) and corpus \(D\), produce a ranked list \(\pi_q\) of up to 1000 documents that are:
1. topically relevant, and
2. humor-compatible.

This is hard because many relevant humorous matches are **lexically indirect** (pun/sound/semantic links instead of exact word overlap).

---

## Slide 2 — End-to-end architecture

Your pipeline is a **multi-stage hybrid ranker**:
1. Lexical retrieval (BM25 + char n-gram TF-IDF + priors)
2. Dense retrieval (embedding similarity)
3. Candidate fusion (RRF)
4. Cross-encoder reranking
5. Humor-aware pair scoring
6. Weighted final fusion + handcrafted features

This design improves recall first, then precision in late ranking.

---

## Slide 3 — Mathematics: BM25 lexical signal

For query term \(t\) and document \(d\):

\[
\text{BM25}(q,d)=\sum_{t\in q} \text{IDF}(t)\cdot \frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}
\]

- \(f(t,d)\): term frequency in \(d\)
- \(k_1,b\): saturation + length normalization parameters

Interpretation: rewards rare, informative query terms while controlling long-document bias.

---

## Slide 4 — Mathematics: character n-gram TF-IDF

For char n-gram vectors \(\mathbf{x}_q,\mathbf{x}_d\):

\[
\text{CharSim}(q,d)=\cos(\mathbf{x}_q,\mathbf{x}_d)=\frac{\mathbf{x}_q\cdot\mathbf{x}_d}{\|\mathbf{x}_q\|\|\mathbf{x}_d\|}
\]

Why useful for JOKER:
- captures orthographic/phonetic-like overlap,
- helps with pun-style lexical distortions,
- complements word-level BM25.

---

## Slide 5 — Dense retrieval and candidate fusion

Dense branch encodes query/doc into vectors \(\mathbf{e}_q,\mathbf{e}_d\), then scores by cosine/dot similarity.

Two ranked lists (lexical, dense) are fused using Reciprocal Rank Fusion:

\[
\text{RRF}(d)=\sum_{r\in R} \frac{1}{k+\text{rank}_r(d)}
\]

- Robust to scale mismatch between rankers
- Increases candidate recall before expensive reranking

---

## Slide 6 — Cross-encoder reranking + humor scorer

### Cross-encoder
A transformer jointly reads \([q;d]\) and outputs relevance score \(s_{ce}(q,d)\).

### Humor pair scorer
A classifier predicts humor compatibility \(s_h(q,d)\).

Both are query-conditioned late signals and typically improve top-rank quality.

---

## Slide 7 — Final weighted fusion equation

For candidate \(d\):

\[
S(d)=w_{lex}s_{lex}(d)+w_{dense}s_{dense}(d)+w_{ce}s_{ce}(d)+w_h s_h(d)+\sum_i w_i f_i(d)
\]

where \(f_i\) are handcrafted features (exact match, overlaps, punctuation/exclamation cues, etc.).

Documents are sorted by \(S(d)\) descending.

---

## Slide 8 — Evaluation metric (MAP@1000)

For each query \(q\), average precision:

\[
AP(q)=\frac{1}{|Rel_q|}\sum_{k=1}^{1000} P@k\cdot \mathbb{1}[d_k\in Rel_q]
\]

Then

\[
MAP@1000=\frac{1}{|Q|}\sum_{q\in Q} AP(q)
\]

Your reported test score: **0.24 MAP@1000**.

---

## Slide 9 — Your model comparison results (train MAP@1000)

Provided results from your pipeline:

| Model | MAP@1000 | Relative rank |
|---|---:|---:|
| roberta-base | **0.287985** | 1 |
| Qwen/Qwen3-Reranker-0.6B | 0.142084 | 2 |
| distilbert-base-uncased | 0.137579 | 3 |
| facebook/bart-base | 0.114561 | 4 |
| camembert-base | 0.111976 | 5 |

Key takeaway: in your setup, **roberta-base is clearly strongest** among tested models.

---

## Slide 10 — Comparison with past CLEF work (train-side evidence)

### CLEF 2024 prior work (University of Amsterdam)
On train data for Task 1 (reported in their notebook):
- `UAms_Task1_Anserini_bm25` → **MAP 0.1582**
- `UAms_Task1_Anserini_rm3` → **MAP 0.3528**
- `UAms_Task1_bm25_CE1000` → **MAP 0.3639**
- `UAms_Task1_rm3_CE1000` → **MAP 0.3682**

### Positioning vs your train runs
- Your best shown model in this experiment: **0.287985** (`roberta-base`).
- This is stronger than plain BM25 baseline above, but below the strongest UAms train configurations reported in that paper.

---

## Slide 11 — Comparison with past CLEF work (test-side context)

- Your pipeline test result: **MAP@1000 = 0.24**.
- CLEF 2025 `pjmathematician` reports English test runs up to **0.3501** MAP@1000.

Interpretation:
- You are in a competitive mid/high range,
- and likely need improvements in late-stage reranking + humor conditioning to close the final gap.

---

## Slide 12 — Error analysis talking points (for viva/presentation Q&A)

1. **Candidate miss errors**: relevant docs absent from fused candidate pool.
2. **Reranker confusion**: topical relevance > humor alignment.
3. **Short query ambiguity**: single-token clues are underspecified.
4. **Feature-weight mismatch**: static weights not optimal across query types.

Suggested next steps:
- query-type-aware fusion weights,
- stronger humor-specific reranker fine-tuning,
- hard-negative mining from dense retrieval.

---

## Slide 13 — Final conclusion slide

- Your work implements a robust **classical + neural hybrid IR pipeline**.
- Mathematical design is principled: retrieval-recall stage + rerank-precision stage.
- Empirically, you already achieved **0.24 MAP@1000 test** and identified a strong local model choice (`roberta-base`) for your setup.
- Clear roadmap remains to approach top CLEF 2025 numbers.

---

## References

1. CLEF 2024 JOKER Task 1 Overview (official):  
   https://ceur-ws.org/Vol-3740/paper-165.pdf
2. University of Amsterdam at CLEF 2024 JOKER (train/test run tables including Task 1 train MAP):  
   https://ceur-ws.org/Vol-3740/paper-181.pdf
3. CLEF 2025 JOKER volume (context):  
   https://ceur-ws.org/Vol-4038/
4. pjmathematician at CLEF 2025 JOKER (train ablations + test MAP@1000):  
   https://ceur-ws.org/Vol-4038/paper_230.pdf
