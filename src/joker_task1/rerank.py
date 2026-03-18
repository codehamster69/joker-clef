from __future__ import annotations

from .retriever import RetrievedDoc


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 8):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def score_pairs(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        model = self._load_model()
        pairs = [(query, doc) for doc in docs]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]

    def rerank(self, query: str, docs: list[tuple[str, str]], top_k: int | None = None) -> list[RetrievedDoc]:
        texts = [text for _, text in docs]
        scores = self.score_pairs(query, texts)
        rows = [RetrievedDoc(docid=docid, score=score) for (docid, _), score in zip(docs, scores)]
        rows.sort(key=lambda row: row.score, reverse=True)
        return rows if top_k is None else rows[:top_k]
