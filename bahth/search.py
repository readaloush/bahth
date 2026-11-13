"""The serving path: an ONNX text encoder and an exact nearest-neighbour index.

Nothing here imports torch. The service needs onnxruntime, tokenizers and numpy only.

The index is a plain matrix product. With tens of thousands of photos an exact search
takes a few milliseconds on one core, so an approximate index would add a dependency
and a recall loss for no gain. `Index.search` is the one place to swap in FAISS or
HNSW when the gallery grows by two orders of magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TextEncoder:
    """Query text -> L2-normalised vector in CLIP's image space."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, max_length: int = 64, threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_length)
        pad = self.tok.token_to_id("[PAD]") or 0
        self.tok.enable_padding(pad_id=pad, pad_token="[PAD]")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            out.append(self.session.run(None, feed)[0])
        v = np.concatenate(out).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)


class Index:
    """Exact cosine search over L2-normalised image embeddings."""

    def __init__(self, embeddings: np.ndarray, photos: list[dict]):
        if len(embeddings) != len(photos):
            raise ValueError(f"{len(embeddings)} embeddings for {len(photos)} photos")
        self.emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.photos = photos

    @classmethod
    def load(cls, folder: str | Path) -> "Index":
        folder = Path(folder)
        return cls(np.load(folder / "gallery.npy"), json.loads((folder / "gallery.json").read_text()))

    def __len__(self) -> int:
        return len(self.photos)

    def search(self, query: np.ndarray, k: int = 12) -> list[dict]:
        k = max(1, min(k, len(self)))
        sim = self.emb @ query.reshape(-1)
        top = np.argpartition(-sim, k - 1)[:k]
        top = top[np.argsort(-sim[top])]
        return [{**self.photos[i], "score": round(float(sim[i]), 4)} for i in top]
