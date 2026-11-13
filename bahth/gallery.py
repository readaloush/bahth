"""Build the photo gallery the search service answers from.

    python -m bahth.gallery --out artifacts

Photos come from the Unsplash Lite dataset (about 25,000 photos, unsplash.com/data).
Its terms do not allow redistributing the images, so only three things are kept:
the CLIP image embedding, the photo id, and what is needed to credit the photographer.
The demo and the service show photos by linking to Unsplash, never from a copy.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LITE = "https://unsplash.com/data/lite/latest"


def read_photos(folder: Path) -> list[dict]:
    """Rows of the photos table, whichever of the two file layouts the release uses."""
    files = sorted(p for p in folder.rglob("photos.*") if p.suffix.startswith((".tsv", ".csv")))
    if not files:
        raise FileNotFoundError(f"no photos table under {folder}")
    csv.field_size_limit(sys.maxsize)
    rows = []
    for f in files:
        with open(f, encoding="utf-8", newline="") as h:
            rows += list(csv.DictReader(h, delimiter="\t" if ".tsv" in f.name else ","))
    return rows


def download_lite(folder: Path) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    if not list(folder.rglob("photos.*")):
        z = folder / "lite.zip"
        req = urllib.request.Request(LITE, headers={"User-Agent": "Mozilla/5.0 bahth"})
        with urllib.request.urlopen(req, timeout=120) as r, open(z, "wb") as h:
            while chunk := r.read(1 << 20):
                h.write(chunk)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder, [n for n in zf.namelist() if Path(n).name.startswith("photos.")])
        z.unlink()
    return read_photos(folder)


def fetch(url: str, width: int = 256, tries: int = 3):
    from PIL import Image
    for _ in range(tries):
        try:
            req = urllib.request.Request(f"{url}?w={width}&q=75&fm=jpg", headers={"User-Agent": "Mozilla/5.0 bahth"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            continue
    return None


def record(row: dict) -> dict:
    name = " ".join(x for x in (row.get("photographer_first_name"), row.get("photographer_last_name")) if x).strip()
    return {"id": row["photo_id"], "image": row["photo_image_url"], "by": name or row.get("photographer_username", ""),
            "user": row.get("photographer_username", ""), "w": int(float(row.get("photo_width") or 0)),
            "h": int(float(row.get("photo_height") or 0)), "blur": row.get("blur_hash", "")}


def build(out: str | Path, limit: int = 0, chunk: int = 512, workers: int = 32, log=print) -> tuple[np.ndarray, list[dict]]:
    import torch
    from sentence_transformers import SentenceTransformer

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = download_lite(out / "unsplash")
    rows = rows[:limit] if limit else rows
    clip = SentenceTransformer("clip-ViT-B-32", device="cuda" if torch.cuda.is_available() else "cpu")
    emb, photos, failed = [], [], 0
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(rows), chunk):
            part = rows[i:i + chunk]
            images = list(pool.map(lambda r: fetch(r["photo_image_url"]), part))
            ok = [(r, im) for r, im in zip(part, images) if im is not None]
            failed += len(part) - len(ok)
            if ok:
                emb.append(clip.encode([im for _, im in ok], batch_size=128, convert_to_numpy=True, normalize_embeddings=True))
                photos += [record(r) for r, _ in ok]
            if (i // chunk) % 8 == 0:
                log(f"gallery: {len(photos)} of {len(rows)} photos embedded, {failed} failed downloads")
    emb = np.concatenate(emb).astype(np.float16)
    np.save(out / "gallery.npy", emb)
    (out / "gallery.json").write_text(json.dumps(photos, ensure_ascii=False))
    log(f"gallery: {len(photos)} photos, {failed} failed downloads, embeddings {emb.shape} float16")
    return emb, photos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.limit)
