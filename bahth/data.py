"""Flickr8k in three languages, keyed by the original image file name.

  images    Flickr8k (Hodosh, Young & Hockenmaier, 2013), 8,091 photos
  English   the original 5 captions per image
  Arabic    3 captions per image, ElJundi et al. (2020), aub-mind/Arabic-Image-Captioning
  Turkish   up to 2 native captions per image, TasvirEt (Unal et al., 2016)

Nothing is redistributed by this repository; `prepare()` downloads from the sources.
The Turkish release on the Hugging Face Hub carries images but not their Flickr file
names, so its rows are matched to the Flickr8k files by image content.
"""
from __future__ import annotations

import json
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES_ZIP = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
ARABIC_REPO = "https://github.com/aub-mind/Arabic-Image-Captioning.git"
TURKISH_HF = "atasoglu/flickr8k-turkish"
LANGS = ("en", "ar", "tr")


def prepare(root: str | Path = "data") -> Path:
    """Downloads images and caption files once. Returns the data root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "Flicker8k_Dataset").exists():
        z = root / "Flickr8k_Dataset.zip"
        if not z.exists():
            print("downloading Flickr8k images (1.1 GB) ...")
            urllib.request.urlretrieve(IMAGES_ZIP, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        z.unlink()
    if not (root / "arabic").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", ARABIC_REPO, str(root / "arabic")], check=True)
    return root


def image_path(root: str | Path, name: str) -> Path:
    return Path(root) / "Flicker8k_Dataset" / name   # (sic) folder name inside the archive


def _read_captions(path: Path) -> dict[str, list[str]]:
    out = defaultdict(list)
    for line in path.read_text(encoding="utf8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and "#" in parts[0] and parts[1].strip():
            out[parts[0].split("#")[0]].append(parts[1].strip())
    return dict(out)


def splits(root: str | Path) -> dict[str, list[str]]:
    """The standard Flickr8k split: 6,000 train, 1,000 dev, 1,000 test images."""
    t = Path(root) / "arabic" / "data" / "Flickr8k_text"
    return {k: (t / f"Flickr_8k.{k}Images.txt").read_text().split() for k in ("train", "dev", "test")}


def _thumb(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def match_turkish(root: str | Path) -> dict[str, list[str]]:
    """Turkish captions keyed by Flickr file name (cached in data/turkish.json).

    Each Hub row is assigned to the Flickr8k file whose 16x16 grayscale thumbnail is
    closest; rows whose best match is not clearly better than the runner-up are dropped
    and counted, so a bad join cannot slip in silently.
    """
    root = Path(root)
    cache = root / "turkish.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf8"))
    from datasets import load_dataset

    names = sorted(p.name for p in (root / "Flicker8k_Dataset").glob("*.jpg"))
    ref = np.stack([_thumb(Image.open(image_path(root, n))) for n in names])
    out, dropped, total = {}, 0, 0
    for split in ("train", "validation", "test"):
        for row in load_dataset(TURKISH_HF, split=split):
            total += 1
            d = ((ref - _thumb(row["image"])) ** 2).mean(1)
            best, second = np.partition(d, 1)[:2]
            if best > 0.05 or best > 0.5 * second:
                dropped += 1
                continue
            caps = [row[c].strip() for c in ("caption0", "caption1") if row.get(c) and row[c].strip()]
            if caps:
                out[names[int(d.argmin())]] = caps
    print(f"Turkish: matched {len(out)} of {total} rows to Flickr8k files, dropped {dropped}")
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
    return out


def captions(root: str | Path) -> dict[str, dict[str, list[str]]]:
    """{'en': {file: [captions]}, 'ar': {...}, 'tr': {...}}"""
    t = Path(root) / "arabic" / "data" / "Flickr8k_text"
    return {"en": _read_captions(t / "Flickr8k.token.txt"), "ar": _read_captions(t / "Flickr8k.arabic.full.txt"), "tr": match_turkish(root)}
