"""Bahth search service.

    BAHTH_ARTIFACTS=artifacts uvicorn service.app:app

GET /search?q=<text>&k=<n>   photos ranked for a query in Arabic, Turkish or English
GET /health                  gallery size and model file in use
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from bahth.search import Index, TextEncoder

ARTIFACTS = Path(os.environ.get("BAHTH_ARTIFACTS", "artifacts"))
MODEL = os.environ.get("BAHTH_MODEL", "text.int8.onnx")

app = FastAPI(title="Bahth", description="Photo search in Arabic, Turkish and English")


@lru_cache(maxsize=1)
def state() -> tuple[TextEncoder, Index]:
    missing = [f for f in (MODEL, "tokenizer.json", "gallery.npy", "gallery.json") if not (ARTIFACTS / f).exists()]
    if missing:
        raise HTTPException(503, f"missing in {ARTIFACTS}: {', '.join(missing)}. Run notebooks/03_index.ipynb to build them.")
    return TextEncoder(ARTIFACTS / MODEL, ARTIFACTS / "tokenizer.json"), Index.load(ARTIFACTS)


@app.get("/health")
def health():
    _, index = state()
    return {"status": "ok", "photos": len(index), "model": MODEL}


@app.get("/search")
def search(q: str = Query(min_length=1, max_length=300), k: int = Query(12, ge=1, le=50)):
    if not q.strip():
        raise HTTPException(422, "The query is empty.")
    encoder, index = state()
    t0 = time.perf_counter()
    vec = encoder([q.strip()])[0]
    t1 = time.perf_counter()
    hits = index.search(vec, k)
    t2 = time.perf_counter()
    for h in hits:
        h["page"] = f"https://unsplash.com/photos/{h['id']}"
    return {"query": q.strip(), "photos": hits, "encode_ms": round((t1 - t0) * 1000, 1), "search_ms": round((t2 - t1) * 1000, 2)}
