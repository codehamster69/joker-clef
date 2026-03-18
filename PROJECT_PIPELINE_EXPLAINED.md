# JOKER CLEF Task 1 Pipeline — A Beginner-to-Advanced Explanation

## Why this document exists

This file explains the **entire project pipeline** in plain language first, and then gradually moves toward a more technical and research-oriented explanation.

If you are completely new to the field, you can think of this project as a system that tries to answer the following question:

> Given a short clue-like query such as a word or concept, how can we find the funniest or most relevant joke-style text snippets from a very large collection?

That is the core problem this repository solves.

---

## 1. The big picture: what is this project doing?

This repository is built for **CLEF JOKER 2025 Task 1**, which is a **humour-aware information retrieval** task.

Let us unpack that slowly.

### 1.1 What is information retrieval?
Information retrieval (IR) is the field behind search engines.

- You type a query.
- The system searches a big document collection.
- It returns a ranked list of documents.

In a normal web search engine, the goal is to find pages that are relevant to your query.

### 1.2 What is different here?
Here, the documents are not ordinary web pages. They are short text snippets that are often joke-like, pun-like, playful, or humor-related.

The queries are also unusual. They are often very short, sometimes just one word like:

- `change`
- `colors`
- `sound`

That means the system cannot rely only on exact matching. It must understand:

- literal word overlap,
- sound/character similarity,
- semantic similarity,
- and whether a document feels humor-compatible for the query.

So this is not “just search.” It is **humour-aware search**.

---

## 2. What data does the project use?

The repository uses four JSON files:

1. **Corpus** — the full set of candidate documents.
2. **Train queries** — example queries for development/training.
3. **Test queries** — the queries we want to rank documents for.
4. **Qrels** — ground-truth relevance labels saying which documents are relevant for which training queries.

### 2.1 Corpus
The corpus contains **77,658 documents**. Each item has:

- a `docid`
- a `text`

Example:

```json
{"docid": "1", "text": "He has a green body, no visible nose, and lives in a trash can."}
```

### 2.2 Train queries
The train set contains **12 training queries**. Example:

```json
{"qid": "8", "query": "colors"}
```

### 2.3 Test queries
The test set contains **219 test queries**. Example:

```json
{"qid": "1", "query": "change"}
```

### 2.4 Qrels
The training qrels contain **660 positive relevance labels** across the training queries.

A qrel entry looks like this:

```json
{"qid": 8, "docid": 151, "qrel": 1}
```

That means document `151` is relevant for query `8`.

---

## 3. What is the repository architecture?

The repository contains a **pipeline**, not just a single model.

That is important.

A pipeline means the system solves the task in several stages, where each stage adds a different kind of intelligence.

At a high level, the project supports **two modes**:

1. **Lexical baseline**
2. **Hybrid neural pipeline**

### 3.1 Lexical baseline
This is the lighter, simpler version.

It uses:
- BM25 word matching
- character n-gram TF-IDF similarity
- a humor prior from qrels
- an exact substring boost

This branch is fast, interpretable, and easy to debug.

### 3.2 Hybrid neural pipeline
This is the stronger, more modern version.

It combines:
- lexical retrieval
- dense retrieval
- fusion of candidate lists
- cross-encoder reranking
- a humor-aware query-document scorer
- handcrafted feature fusion

This is the repository’s main advanced system.

---

## 4. Beginner explanation: how the pipeline works like a real-world search team

Imagine you have a huge box of 77,658 joke snippets and someone gives you a query like `change`.

Instead of asking one person to search the whole box, this project uses a **team of specialists**.

### Specialist 1: the keyword matcher
This person looks for obvious word overlap.
If the query word appears in the document, that is a strong clue.

### Specialist 2: the spelling/sound pattern matcher
This person looks for similar character patterns.
That matters for humor because jokes and puns often depend on letter-level or sound-like similarity.

### Specialist 3: the semantic meaning matcher
This person looks for meaning-level similarity even when the same words are not used.

### Specialist 4: the close reader
This person takes only the best candidate documents and reads them more carefully together with the query.

### Specialist 5: the humor judge
This person asks: “Even if this is relevant, is it the right kind of humorous match for this query?”

### Specialist 6: the final decision maker
This person combines the opinions of all the specialists and produces the final ranked list.

That is basically what this repository does.

---

## 5. Step-by-step explanation of the current pipeline

Now let us walk through the actual project flow.

## Step 0: Load the data
The helper utilities read JSON input files and convert them into Python data structures.

The repository also provides helpers to:
- map document IDs to document rows,
- map query IDs to query rows,
- convert qrels into a per-query relevant-document map,
- and write outputs back to JSON or ZIP.

This is the plumbing that makes the rest of the pipeline possible.

---

## Step 1: Build the lexical baseline index

The first true ranking component is `HybridTask1Retriever`, which is actually a lexical retriever despite the class name.

It performs several preprocessing operations over the full document collection.

### 5.1 Tokenization
Each document is split into lowercase word tokens.

Example:

- Text: `Why did the tomato blush?`
- Tokens: `why`, `did`, `the`, `tomato`, `blush`

These tokens are used for BM25-style scoring.

### 5.2 Character n-grams
The retriever also creates character n-grams from lengths 3 to 5.

Why is this useful?
Because humor, puns, and playful language often involve:
- partial word matches,
- strange spelling overlap,
- near-rhymes,
- stylistic fragments.

Character-level similarity can catch relationships that plain word overlap misses.

### 5.3 BM25 statistics
The system computes:
- term frequencies per document,
- document lengths,
- document frequencies,
- inverse document frequency values,
- average document length.

This is standard classical IR machinery.

### 5.4 Character TF-IDF statistics
For character n-grams, the system also computes:
- n-gram frequencies,
- n-gram document frequencies,
- n-gram inverse document frequencies,
- vector norms for cosine similarity.

### 5.5 Humor prior from qrels
If training qrels are available, every document that appears as relevant in training gets a humor prior of `1.0`; other documents get `0.0`.

This is a very simple heuristic saying:

> “Documents that were judged relevant before may be generally more humor-rich or more task-compatible.”

This is not query-specific. It is just a global prior.

---

## Step 2: Run lexical scoring for a query

When a user query comes in, the lexical retriever computes four signals for every document.

### 5.6 Signal A: BM25 word score
BM25 measures how strongly the query words match the document words, while balancing term rarity and document length.

Intuition:
- Matching a rare useful word is better than matching a very common word.
- Matching the word multiple times helps, but with diminishing returns.
- Very long documents are penalized somewhat.

### 5.7 Signal B: character TF-IDF cosine similarity
This checks how similar the query and document are at the character-fragment level.

This is especially meaningful in humor tasks because the relation between query and joke may be closer in sound/letter pattern than in exact wording.

### 5.8 Signal C: humor prior
This is the qrel-derived prior discussed above.

### 5.9 Signal D: exact match boost
If the lowercased query appears directly inside the lowercased document text, the document gets an extra boost.

### 5.10 Lexical final score
The lexical branch combines all four signals using weighted addition:

- BM25 weight
- character similarity weight
- humor prior weight
- exact match boost weight

Documents with positive scores are sorted, and the top results are returned.

This gives the project a **strong classical first-stage retriever**.

---

## Step 3: Optional auto-tuning of lexical parameters

The repository can automatically search over combinations of lexical parameters.

It tries different values for:
- `k1`
- `b`
- `char_weight`
- `humor_weight`
- `match_boost`

To do that, it splits the training qrels by query into:
- a train portion,
- a validation portion.

Then it picks the parameter set with the best **MAP@K**.

### What is MAP@K?
MAP means **Mean Average Precision**.

A beginner-friendly interpretation:
- If relevant documents appear high in the ranked list, the score is good.
- If they appear low, the score is worse.
- The system averages this quality across queries.

So MAP@K is not just about whether the right document appears somewhere. It rewards ranking relevant items early.

---

## Step 4: Build the dense retrieval branch

The hybrid pipeline adds a second kind of retriever called a **dense retriever**.

### 5.11 What is dense retrieval?
Instead of matching words directly, dense retrieval turns texts into vectors of numbers called **embeddings**.

If two texts have similar meaning, their vectors should be close in this embedding space.

### 5.12 What the repository does
The `DenseRetriever` uses a sentence-transformer encoder, by default:

- `BAAI/bge-small-en-v1.5`

It encodes each document into an embedding vector and stores:
- `embeddings.npy`
- `docids.json`
- `meta.json`
- `faiss.index` when FAISS is available

### 5.13 Why this matters
Dense retrieval helps when query and document are related in meaning but do **not** share the same exact words.

That is crucial because humorous text often works through indirection, paraphrase, or concept-level association.

### 5.14 Retrieval speed
To search efficiently, the project uses:
- FAISS if available,
- otherwise a NumPy dot-product fallback.

So the branch is both stronger semantically and still practical to run locally.

---

## Step 5: Fuse lexical and dense candidates

After getting results from both branches:
- lexical retrieval
- dense retrieval

The project merges them.

### 5.15 Why fusion is needed
Lexical and dense retrieval are good at different things.

- Lexical retrieval is good at exact overlap and surface-form clues.
- Dense retrieval is good at semantic matching.

If we trust only one, we lose useful information.

### 5.16 Reciprocal Rank Fusion (RRF)
The repository first uses **reciprocal rank fusion**.

RRF gives credit to a document if it appears high in either ranking list.
A document that appears near the top of both lists gets even more credit.

Why RRF is good:
- simple,
- robust,
- does not require learning,
- works well for combining different retrieval styles.

### 5.17 Candidate set construction
The project then builds a merged candidate dictionary where each document may carry:
- lexical score,
- dense score,
- reranker score,
- humor score,
- handcrafted feature scores,
- final score.

At this point, we now have a shared pool of candidate documents that can be refined further.

---

## Step 6: Add handcrafted humor-related features

For top candidates, the project computes extra surface-level features.

These include:
- exact query match,
- token overlap,
- character overlap,
- normalized document length,
- punctuation density,
- exclamation density,
- quote density,
- repeated-word frequency.

### Why these features exist
Humor often leaves stylistic traces.
For example, funny text may:
- use punctuation expressively,
- include repetition,
- use quotation marks,
- contain strong overlap with a clue word,
- or have certain compact lengths.

These features are not magic by themselves, but they provide extra useful signals for ranking.

---

## Step 7: Cross-encoder reranking

Now the pipeline gets more advanced.

After fusion, the system takes only the strongest candidates and sends them to a **cross-encoder reranker**.

By default the project uses:

- `cross-encoder/ms-marco-MiniLM-L12-v2`

### 5.18 What is a reranker?
A reranker does not search the whole corpus from scratch.
Instead, it reads:
- one query
- one candidate document
- together

Then it gives a more informed relevance score.

### 5.19 Why reranking is useful
First-stage retrieval must be fast, so it uses relatively cheap scoring.
A reranker is slower but smarter.

That is why the system uses it only on the top candidate subset, not on all 77,658 documents.

### 5.20 Practical role in this repository
The reranker improves ordering among already promising candidates.
This often leads to better final precision.

---

## Step 8: Humor-aware query-document scoring

This is one of the most task-specific parts of the repository.

### 5.21 What is trained?
The project can train a compact query-conditioned humor/relevance scorer.

By default it fine-tunes:
- `roberta-base`

### 5.22 Training data construction
The training process creates pairs of:
- query text,
- document text,
- binary label.

Positive pairs come from qrels.

Negative pairs are built using **hard negatives**:
- the lexical retriever fetches high-ranking but non-relevant documents,
- random negatives are added if needed.

This is smart because hard negatives teach the model to distinguish between:
- documents that look superficially plausible,
- and documents that are truly relevant/humor-compatible.

### 5.23 How inference works
At prediction time, the trained model scores query-document pairs and outputs a probability-like humor/relevance score.

This score is then attached to the candidate document.

### 5.24 Why this is better than the lexical humor prior
The lexical prior is global and crude.
This classifier is **query-conditioned**.

So instead of asking:
> “Is this document generally humor-ish?”

it asks:
> “Is this document a good humorous match for this specific query?”

That is a major conceptual improvement.

---

## Step 9: Final weighted fusion

Once all the available signals are attached to each candidate, the project performs a final weighted fusion.

The default weights combine:
- lexical score,
- dense score,
- reranker score,
- humor classifier score,
- handcrafted feature contributions.

The candidate’s final score is a weighted sum of all of these parts.

Then all candidates are sorted again and the top `K` documents are returned.

This final ranked list is what becomes the system output.

---

## Step 10: Save predictions and optionally ZIP them

The repository writes predictions as JSON rows containing:
- `run_id`
- `manual`
- `qid`
- `docid`
- `rank`
- `score`

It can also package the prediction file into a submission ZIP.

That makes the system ready for shared-task submission.

---

## 6. What commands exist in the project?

The project exposes its functionality through a CLI and a GUI.

## 6.1 CLI commands

### `predict`
Runs the original lexical baseline.

### `build-dense-index`
Builds the embedding index for dense retrieval.

### `train-humor`
Trains the query-conditioned humor pair classifier.

### `predict-hybrid`
Runs the full hybrid pipeline.

### `ablate`
Runs multiple versions of the system so you can compare components.

### `eval`
Computes MAP@K for predictions against qrels.

---

## 6.2 GUI

The GUI is a practical control panel that lets the user:
- choose baseline vs hybrid mode,
- configure models,
- build the dense index,
- train the humor model,
- run prediction,
- evaluate results,
- auto-tune lexical settings,
- watch logs and progress,
- and monitor CPU/RAM/GPU usage.

So the project is not only a research pipeline but also a fairly usable experimentation environment.

---

## 7. The current repository approach in one sentence

If we compress the whole design into one sentence, it is this:

> Start with a strong lexical search engine, add a semantic retriever, merge their candidates, refine them with a deep pairwise reranker, add a query-aware humor model plus surface humor features, and then fuse everything into a final ranking.

That is the current approach.

---

## 8. Why this design makes sense for humor retrieval

Humor retrieval is hard because relevance can come from different kinds of relationships.

A document may be relevant because:
- it literally contains the query word,
- it contains a wordplay variant,
- it is semantically connected but lexically different,
- it has joke-like style,
- it matches the query’s intended humorous framing.

No single method captures all of these equally well.

That is why the project uses a **multi-signal ranking architecture**.

You can think of the architecture as balancing four types of intelligence:

1. **Exact lexical matching**
2. **Subword/character similarity**
3. **Semantic meaning matching**
4. **Query-aware humor judgment**

This is a very sensible design for the task.

---

## 9. Comparison with older or past CLEF-style approaches

This section is based on the repository’s own comparison framing and how the implemented system is positioned relative to earlier families of approaches.

## 9.1 Original repository baseline

### What it was
The original baseline is the lightweight lexical pipeline:
- BM25
- character 3–5 gram TF-IDF
- humor prior from train qrels
- exact match boost

### What it does well
- fast,
- cheap,
- easy to understand,
- surprisingly strong for lexical clues.

### What it misses
- meaning-level matching is limited,
- humor modeling is very crude,
- there is no powerful neural reranking,
- no query-conditioned humor understanding.

### Compared to the current pipeline
The current system keeps this baseline as a strong first stage but then **extends it upward** rather than replacing it entirely.

That is a good engineering decision because the lexical baseline remains useful even after neural components are added.

---

## 9.2 Classical IR approaches such as BM25 / Indri / RM3

### What these systems focus on
Older classical IR methods emphasize:
- lexical evidence,
- probabilistic term weighting,
- structured retrieval,
- query expansion or pseudo-relevance feedback.

### Strengths
- stable,
- interpretable,
- strong at first-stage retrieval,
- often hard to beat as a baseline.

### Weaknesses for this task
Humor relevance is often indirect.
So a purely lexical system can fail when:
- the joke and the query share little vocabulary,
- the relation is semantic rather than literal,
- the key signal is style or wordplay structure.

### Compared to the current pipeline
The current repository is stronger because it adds:
- dense semantic retrieval,
- reranking,
- and query-conditioned humor scoring.

So compared with classical-only pipelines, this project is much more capable of handling non-literal matches.

---

## 9.3 CLEF 2024-style two-stage systems

### Typical pattern
A common recent pattern is:
1. retrieve with BM25 or another lexical method,
2. apply a neural model as a filter or reranker.

### Why that helped historically
This was a natural upgrade path from classical IR to neural IR.
It improved quality while keeping the system manageable.

### Limitations
If the first-stage lexical retriever misses semantically distant but relevant documents, the downstream neural model never gets a chance to see them.

### Compared to the current pipeline
The current repository improves on that design by **adding a dense branch before reranking**.
That means the candidate pool is broader and more semantically informed from the start.

This is a major step forward.

---

## 9.4 CLEF 2025 modular hybrid approaches

### Typical pattern
The repository’s comparison document describes a family of competitive 2025 systems that combine:
- lexical retrieval,
- semantic retrieval,
- reranking,
- and wordplay-oriented or humor-aware features.

### Why this matters
This is the closest family to the current project.

### Compared to the current pipeline
The current repository clearly belongs to this modern modular-hybrid family.
Its implementation is practical and ablation-friendly:
- lexical branch,
- dense branch,
- fusion,
- reranking,
- humor scoring,
- handcrafted features.

This means the project is aligned with a strong contemporary research direction rather than using a dated architecture.

---

## 9.5 Relevance-aware classification approaches

### Typical pattern
Another modern design is to use:
- a semantic retriever,
- plus a query-document classifier to judge relevance or humor compatibility.

### Why this is important
That idea appears directly inside this repository through the `HumorPairScorer`.

### Compared to the current pipeline
The project goes one step further than a classifier-only approach by not relying on semantic classification alone.
It keeps:
- lexical cues,
- dense retrieval,
- reranking,
- feature fusion.

So the current system is more robust than a design that leans only on a neural pair classifier.

---

## 9.6 Very large-model pipelines

### Typical pattern
Some high-end future-facing systems use:
- larger embedding models,
- larger rerankers,
- LLM-style filtering or reasoning,
- or explanation-based selection.

### Strengths
These may have the highest quality ceiling.

### Weaknesses
They are:
- expensive,
- slower,
- harder to train and deploy,
- and less practical on modest hardware.

### Compared to the current pipeline
The current repository intentionally sits in the **practical sweet spot**:
- much stronger than lexical-only systems,
- more realistic than giant LLM-heavy pipelines,
- suitable for local experimentation.

That is one of its best design qualities.

---

## 10. What is technically elegant about the current project?

Several choices in this repository are especially well judged.

### 10.1 It does not throw away the baseline
Instead of abandoning lexical retrieval, it preserves it as a valuable branch.

That is correct, because humor retrieval still benefits from lexical and orthographic clues.

### 10.2 It uses candidate generation plus refinement
This is the standard scalable retrieval pattern:
- cheap first-stage retrieval,
- expensive second-stage refinement.

That is far better than trying to run heavy models over the whole corpus.

### 10.3 It mixes learned and engineered signals
This is useful because humor is messy.
Not all useful cues are captured automatically by a neural model, especially in small-data settings.

### 10.4 It supports ablation
A research pipeline should let you ask:
- What does dense retrieval add?
- What does reranking add?
- What does the humor model add?

This repository supports that explicitly.

### 10.5 It is usable, not just theoretical
The GUI, CLI, training support, evaluation support, and saved artifacts make the repository a workflow system, not just a code dump.

---

## 11. The main limitations of the current approach

Even though the current pipeline is strong, it still has meaningful limitations.

## 11.1 Tiny labeled training query set
There are only **12 training queries** in the train query file.

That is very small for training a robust humor-aware system.

Implications:
- the humor model may overfit,
- auto-tuning estimates may be unstable,
- the qrel-derived humor prior is coarse,
- generalization can be fragile.

## 11.2 Humor prior is document-level and global
The lexical humor prior says a document was relevant before, so maybe it is generally useful again.

That can help, but it is not a true model of humor and not truly query-specific.

## 11.3 The reranker is relevance-oriented, not humor-specialized
The default cross-encoder comes from a standard passage-ranking setup, not a humor-specific one.

That means it may improve relevance ordering but not necessarily capture subtle comedic structure.

## 11.4 Handcrafted features are simple
The handcrafted features are useful, but they are basic stylistic cues.
They do not explicitly model:
- incongruity,
- ambiguity,
- phonetic punning,
- double meanings,
- cultural references,
- setup-punchline structure.

## 11.5 Candidate quality still matters a lot
If the relevant document is not retrieved by lexical or dense search, neither reranking nor humor scoring can recover it.

So first-stage recall remains critical.

## 11.6 The current training negatives come mostly from lexical hard negatives
That is good, but future versions could benefit from harder semantic negatives or adversarial negatives produced by dense retrieval or other models.

---

## 12. Future scope: where this project can go next

Now let us discuss the future in a realistic research-and-engineering sense.

## 12.1 Better dense retrieval
The dense branch could improve through:
- stronger encoders,
- domain-adapted encoders,
- multilingual encoders,
- contrastive fine-tuning on task-specific query-document pairs.

This would increase semantic recall.

## 12.2 Better reranking
The reranker could be upgraded by:
- using a stronger cross-encoder,
- fine-tuning it on CLEF JOKER pairs,
- training it jointly for humor-aware relevance rather than generic relevance.

That would make second-stage ranking more task-aligned.

## 12.3 Better humor modeling
A major future direction is richer humor understanding.
Possible additions include:
- pun detection,
- phonetic similarity features,
- ambiguity signals,
- incongruity modeling,
- figurative language detection,
- joke structure modeling.

This is probably one of the most valuable long-term research directions.

## 12.4 Learn the fusion weights instead of hand-setting them
Currently the final fusion uses manually defined default weights.

A future system could learn fusion weights from validation data or with a learning-to-rank model.
Examples:
- LambdaMART-style ranking,
- logistic regression over features,
- neural fusion layers,
- query-dependent gating.

That would make the combination smarter and less heuristic.

## 12.5 Use query expansion and reformulation
For extremely short queries, the system could automatically generate expansions such as:
- synonyms,
- related concepts,
- pun variants,
- phonetic variants,
- explanation-style expansions from language models.

This could improve both lexical and dense retrieval.

## 12.6 Better negative sampling for the humor classifier
Future training data could include:
- dense hard negatives,
- reranker-confusing negatives,
- near-miss pun candidates,
- cross-query distractors.

This would likely sharpen the classifier’s discrimination ability.

## 12.7 Multilingual or cross-lingual extension
The current defaults are English-centric.
Future work could expand to multilingual humor retrieval with:
- multilingual embeddings,
- multilingual rerankers,
- language-specific humor features,
- translation-aware retrieval.

## 12.8 Retrieval-augmented reasoning with larger models
A future system could let a language model examine top candidates and reason about:
- why a result is funny,
- what kind of wordplay it uses,
- whether it matches the intended sense of the query.

This is promising, but it must be balanced against compute cost and reproducibility.

## 12.9 Better evaluation beyond MAP
MAP is useful, but future experiments may also want:
- recall-oriented measures,
- nDCG,
- manual error analysis,
- humor-type breakdowns,
- per-query failure categorization.

That would lead to deeper system understanding.

## 12.10 End-to-end learned retrieval
A more ambitious future path is to train the dense retriever, reranker, and humor scorer in a more unified way.

That would move the project from a modular pipeline to a more end-to-end optimized system.

The tradeoff is that end-to-end systems are usually harder to debug and require more data.

---

## 13. Practical research roadmap for this repository

If I were prioritizing future work for this exact repository, I would suggest this order:

### Short-term improvements
1. Add stronger ablation reporting.
2. Tune fusion weights more systematically.
3. Add dense-hard negatives to humor training.
4. Compare multiple dense encoders and rerankers.
5. Save more diagnostic outputs per query.

### Mid-term improvements
1. Fine-tune the reranker for this task.
2. Train a better query-conditioned humor model.
3. Add phonetic and pun-aware features.
4. Add query expansion for short clue words.

### Long-term improvements
1. Learn fusion instead of hand-designing it.
2. Build humor-type-aware ranking.
3. Add multilingual support.
4. Explore lightweight LLM-assisted reranking or explanation-based scoring.

---

## 14. One more very simple summary for a total beginner

If someone has **zero knowledge** of the field, the simplest explanation is this:

- The project is a smart search engine for humorous text.
- It does not trust just one search method.
- First it searches by words and letter patterns.
- Then it also searches by meaning using embeddings.
- Then it merges the candidate results.
- Then it rereads the best candidates with stronger models.
- Then it uses a humor-aware classifier to judge whether a query and document belong together.
- Then it mixes all signals and returns the final ranking.

Older approaches mostly relied more on keyword matching and simpler reranking.
The current approach is better because it combines:
- classical search,
- neural search,
- pairwise deep scoring,
- and humor-aware features.

Its future lies in becoming more:
- task-specific,
- learned,
- semantically rich,
- and capable of understanding actual humor mechanisms instead of just surface relevance.

---

## 15. Final conclusion

This repository represents a **modern practical hybrid retrieval system** for humor-aware search.

Its core strengths are:
- a strong lexical base,
- semantic dense retrieval,
- modular candidate fusion,
- neural reranking,
- query-conditioned humor scoring,
- and an experimentation-friendly workflow.

Compared with older CLEF-style or classical lexical approaches, it is more expressive, more semantically capable, and more aligned with the difficulty of the task.

Compared with very large-model pipelines, it is much more practical and reproducible.

So the current project is best understood as a **balanced research pipeline**:
strong enough to be competitive, modular enough to analyze, and lightweight enough to improve iteratively.

