from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .retriever import RetrievedDoc

ProgressFn = Callable[[str, float], None]


@dataclass(frozen=True)
class DenseIndexArtifacts:
    index_dir: Path
    embeddings_path: Path
    docids_path: Path
    meta_path: Path
    faiss_path: Path

    @classmethod
    def for_dir(cls, index_dir: str | Path) -> "DenseIndexArtifacts":
        root = Path(index_dir)
        return cls(
            index_dir=root,
            embeddings_path=root / "embeddings.npy",
            docids_path=root / "docids.json",
            meta_path=root / "meta.json",
            faiss_path=root / "faiss.index",
        )


class DenseEncoder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32, normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode_texts(self, texts: Iterable[str], progress: ProgressFn | None = None, progress_span: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        rows = list(texts)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load_model()
        start_pct, end_pct = progress_span
        chunks: list[np.ndarray] = []
        total = len(rows)
        for start in range(0, total, self.batch_size):
            batch = rows[start : start + self.batch_size]
            emb = model.encode(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            chunks.append(np.asarray(emb, dtype=np.float32))
            if progress:
                done = min(total, start + len(batch))
                frac = done / total
                progress(
                    f"Dense encoding: {done}/{total}",
                    start_pct + (end_pct - start_pct) * frac,
                )
        return np.vstack(chunks)


class DenseIndex:
    def __init__(self, artifacts: DenseIndexArtifacts):
        self.artifacts = artifacts

    def save(self, embeddings: np.ndarray, docids: list[str], meta: dict, progress: ProgressFn | None = None) -> None:
        self.artifacts.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.artifacts.embeddings_path, embeddings.astype(np.float32))
        self.artifacts.docids_path.write_text(json.dumps(docids, ensure_ascii=False, indent=2), encoding="utf-8")
        self.artifacts.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress("Dense index artifacts saved.", 0.92)
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, str(self.artifacts.faiss_path))
            if progress:
                progress("FAISS index written.", 1.0)
        except Exception:
            if progress:
                progress("FAISS unavailable; using numpy fallback retrieval.", 1.0)

    def load(self) -> tuple[np.ndarray, list[str], dict]:
        embeddings = np.load(self.artifacts.embeddings_path)
        docids = json.loads(self.artifacts.docids_path.read_text(encoding="utf-8"))
        meta = json.loads(self.artifacts.meta_path.read_text(encoding="utf-8"))
        return np.asarray(embeddings, dtype=np.float32), list(docids), dict(meta)


class DenseRetriever:
    def __init__(self, model_name: str, index_dir: str | Path, device: str | None = None, batch_size: int = 32):
        self.model_name = model_name
        self.artifacts = DenseIndexArtifacts.for_dir(index_dir)
        self.encoder = DenseEncoder(model_name=model_name, device=device, batch_size=batch_size)
        self.embeddings: np.ndarray | None = None
        self.docids: list[str] = []
        self.meta: dict = {}
        self._faiss_index = None

    def build(self, docs: Iterable[dict], progress: ProgressFn | None = None) -> None:
        rows = list(docs)
        texts = [str(row["text"]) for row in rows]
        docids = [str(row["docid"]) for row in rows]
        if progress:
            progress(f"Preparing dense index for {len(docids)} documents...", 0.02)
        embeddings = self.encoder.encode_texts(texts, progress=progress, progress_span=(0.08, 0.88))
        meta = {"model_name": self.model_name, "size": len(docids), "dim": int(embeddings.shape[1]) if len(docids) else 0}
        DenseIndex(self.artifacts).save(embeddings=embeddings, docids=docids, meta=meta, progress=progress)
        self.embeddings = embeddings
        self.docids = docids
        self.meta = meta
        self._load_faiss()

    def load(self) -> None:
        embeddings, docids, meta = DenseIndex(self.artifacts).load()
        self.embeddings = embeddings
        self.docids = docids
        self.meta = meta
        self._load_faiss()

    def ensure_ready(self, docs: Iterable[dict] | None = None, progress: ProgressFn | None = None) -> None:
        if self.embeddings is not None and self.docids:
            return
        if self.artifacts.embeddings_path.exists() and self.artifacts.docids_path.exists() and self.artifacts.meta_path.exists():
            if progress:
                progress(f"Loading dense index from {self.artifacts.index_dir}", 0.14)
            self.load()
            return
        if docs is None:
            raise FileNotFoundError(f"Dense index not found in {self.artifacts.index_dir}")
        self.build(docs, progress=progress)

    def _load_faiss(self) -> None:
        if not self.artifacts.faiss_path.exists():
            self._faiss_index = None
            return
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(self.artifacts.faiss_path))
        except Exception:
            self._faiss_index = None

    def rank(self, query: str, top_k: int = 1000) -> list[RetrievedDoc]:
        if self.embeddings is None:
            raise RuntimeError("Dense index is not loaded.")
        q_emb = self.encoder.encode_texts([query])
        query_vec = np.asarray(q_emb[0], dtype=np.float32)
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec[None, :], min(top_k, len(self.docids)))
            idxs = indices[0].tolist()
            vals = scores[0].tolist()
        else:
            vals_arr = self.embeddings @ query_vec
            if top_k >= len(vals_arr):
                idxs = np.argsort(-vals_arr).tolist()
            else:
                idxs = np.argpartition(-vals_arr, top_k - 1)[:top_k].tolist()
                idxs.sort(key=lambda i: float(vals_arr[i]), reverse=True)
            vals = [float(vals_arr[i]) for i in idxs]
        rows: list[RetrievedDoc] = []
        for idx, score in zip(idxs, vals):
            if idx < 0:
                continue
            rows.append(RetrievedDoc(docid=self.docids[idx], score=float(score)))
        return rows[:top_k]
