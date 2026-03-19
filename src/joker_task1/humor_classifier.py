from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import docs_by_id, queries_by_id
from .retriever import HybridTask1Retriever

ProgressFn = Callable[[str, float], None]


@dataclass(frozen=True)
class PairExample:
    query: str
    doc_text: str
    label: float


class PairDataset(Dataset):
    def __init__(self, examples: list[PairExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.query,
            ex.doc_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(float(ex.label), dtype=torch.float32)
        return item


class HumorPairScorer:
    def __init__(self, model_dir: str | Path, device: str | None = None, max_length: int = 256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = str(model_dir)
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, num_labels=1)
        self.device = torch.device(self.device_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, docs: list[str], batch_size: int = 8) -> list[float]:
        rows: list[float] = []
        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start : start + batch_size]
            enc = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
                probs = torch.sigmoid(logits).detach().cpu().tolist()
            if isinstance(probs, float):
                rows.append(float(probs))
            else:
                rows.extend(float(p) for p in probs)
        return rows


def build_pair_examples(
    docs: Iterable[dict],
    queries: Iterable[dict],
    qrels: Iterable[dict],
    negatives_per_positive: int = 3,
    seed: int = 13,
    progress: ProgressFn | None = None,
) -> list[PairExample]:
    doc_map = docs_by_id(docs)
    query_map = queries_by_id(queries)
    positives: list[PairExample] = []
    qid_to_pos_docids: dict[str, set[str]] = {}
    for row in qrels:
        qid = str(row["qid"])
        docid = str(row["docid"])
        if int(row.get("qrel", 0)) <= 0:
            continue
        qid_to_pos_docids.setdefault(qid, set()).add(docid)
        positives.append(PairExample(query=query_map[qid]["query"], doc_text=doc_map[docid]["text"], label=1.0))
    if progress:
        progress(f"Collected {len(positives)} positive training pairs.", 0.08)

    rnd = random.Random(seed)
    all_docids = list(doc_map.keys())
    retriever = HybridTask1Retriever()
    retriever.fit(docs)

    negatives: list[PairExample] = []
    total_qids = max(1, len(qid_to_pos_docids))
    for idx, (qid, pos_docids) in enumerate(qid_to_pos_docids.items(), start=1):
        query_text = str(query_map[qid]["query"])
        hard_pool = [row.docid for row in retriever.rank(query_text, top_k=max(50, negatives_per_positive * len(pos_docids) * 2))]
        hard_pool = [docid for docid in hard_pool if docid not in pos_docids]
        while len(hard_pool) < negatives_per_positive * len(pos_docids):
            candidate = rnd.choice(all_docids)
            if candidate not in pos_docids:
                hard_pool.append(candidate)
        for docid in hard_pool[: negatives_per_positive * len(pos_docids)]:
            negatives.append(PairExample(query=query_text, doc_text=doc_map[docid]["text"], label=0.0))
        if progress:
            progress(f"Built hard negatives: {idx}/{total_qids} queries", 0.08 + 0.17 * (idx / total_qids))

    combined = positives + negatives
    rnd.shuffle(combined)
    return combined


def train_humor_pair_classifier(
    docs: Iterable[dict],
    queries: Iterable[dict],
    qrels: Iterable[dict],
    output_dir: str | Path,
    model_name: str = "roberta-base",
    device: str | None = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    negatives_per_positive: int = 3,
    seed: int = 13,
    progress: ProgressFn | None = None,
) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Preparing humor training data with model {model_name}", 0.02)

    examples = build_pair_examples(
        docs,
        queries,
        qrels,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
        progress=progress,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    dataset = PairDataset(examples, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    model_device = torch.device(device_name)
    model.to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    losses: list[float] = []
    total_steps = max(1, len(loader))
    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        if progress:
            progress(f"Training humor model: epoch {epoch + 1}/{epochs}", 0.28 + 0.68 * (epoch / max(epochs, 1)))
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(model_device)
            batch = {k: v.to(model_device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
            if progress and (step % 5 == 0 or step == total_steps):
                done = (epoch + (step / total_steps)) / max(epochs, 1)
                progress(
                    f"Training humor model: epoch {epoch + 1}/{epochs}, step {step}/{total_steps}, loss={loss.item():.4f}",
                    0.28 + 0.68 * done,
                )
        losses.append(epoch_loss / max(steps, 1))

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metrics = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "negatives_per_positive": negatives_per_positive,
        "train_loss": losses[-1] if losses else math.nan,
        "examples": len(examples),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(f"Saved humor model to {out_dir}", 1.0)
    return metrics
