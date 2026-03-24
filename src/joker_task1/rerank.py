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
            tokenizer = getattr(self._model, "tokenizer", None)
            model = getattr(self._model, "model", None)
            if tokenizer is not None and tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.sep_token is not None:
                    tokenizer.pad_token = tokenizer.sep_token
                elif tokenizer.cls_token is not None:
                    tokenizer.pad_token = tokenizer.cls_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
            if tokenizer is not None and model is not None and getattr(model.config, "pad_token_id", None) is None:
                if getattr(tokenizer, "pad_token_id", None) is not None:
                    model.config.pad_token_id = tokenizer.pad_token_id
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
