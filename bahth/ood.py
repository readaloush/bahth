"""Out-of-domain data: the XTD test set and a parallel English-Arabic caption corpus.

XTD (Aggarwal & Kale, 2020) is 1,000 MS-COCO photos with one human-written caption per
language; the Arabic captions come from the XTD-11 extension (Almubarak et al.). None of
these photos or captions are used for training, so XTD measures what fine-tuning on
Flickr8k does to pictures of a different kind.

The parallel corpus is the Arabic translation of MS-COCO captions released with AraCLIP
(Al-Barham et al., 2024). Captions of the 1,000 XTD photos are removed from it, otherwise
the test set would leak into training.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

XTD_REPO = "https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10.git"
XTD_AR = ("khalidalt/xtd_11", "test/ar.json")
COCO_AR = ("Arabic-Clip/Arabic_MSCOCO_training", "Arabic_MSCOCO_training.csv")
COCO_URL = "http://images.cocodataset.org/{split}/{name}"


def coco_id(name: str) -> int:
    return int(re.search(r"(\d+)\.jpg$", name).group(1))


def _fetch(args):
    name, dst = args
    if dst.exists() and dst.stat().st_size > 0:
        return True
    split = "train2014" if "train2014" in name else "val2014"
    for _ in range(3):
        try:
            urllib.request.urlretrieve(COCO_URL.format(split=split, name=name), dst)
            return True
        except Exception:
            pass
    return False


def prepare_xtd(root: str | Path = "data"):
    """Returns (names, {'en': {...}, 'ar': {...}, 'tr': {...}}, image_dir). Only photos that downloaded are kept."""
    from huggingface_hub import hf_hub_download

    root = Path(root)
    if not (root / "xtd").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", XTD_REPO, str(root / "xtd")], check=True)
    d = root / "xtd" / "XTD10"
    names = d.joinpath("test_image_names.txt").read_text().split()
    caps = {}
    for lang in ("en", "tr"):
        lines = d.joinpath(f"test_1kcaptions_{lang}.txt").read_text(encoding="utf8").split("\n")[:len(names)]
        caps[lang] = {n: [c.strip()] for n, c in zip(names, lines) if c.strip()}
    raw = Path(hf_hub_download(XTD_AR[0], XTD_AR[1], repo_type="dataset")).read_text(encoding="utf8")
    dec, pos, ar = json.JSONDecoder(), 0, {}
    while True:                                   # the file is a run of JSON objects, not valid JSON as a whole
        m = re.compile(r"\{").search(raw, pos)
        if not m:
            break
        obj, pos = dec.raw_decode(raw, m.start())
        if isinstance(obj, dict) and obj.get("image_name") and obj.get("text"):
            ar[obj["image_name"]] = [obj["text"].strip()]
    caps["ar"] = {n: ar[n] for n in names if n in ar}

    img_dir = root / "xtd_images"
    img_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(_fetch, [(n, img_dir / n) for n in names]))
    kept = [n for n, o in zip(names, ok) if o]
    print(f"XTD: {len(kept)} of {len(names)} photos; captions en {len(caps['en'])}, ar {len(caps['ar'])}, tr {len(caps['tr'])}")
    return kept, caps, img_dir


def coco_parallel(exclude_names: list[str], limit: int | None = None) -> list[tuple[str, str]]:
    """(English, Arabic) caption pairs from Arabic MS-COCO, without the XTD photos."""
    from huggingface_hub import hf_hub_download

    banned = {coco_id(n) for n in exclude_names}
    pairs, dropped = [], 0
    with open(hf_hub_download(COCO_AR[0], COCO_AR[1], repo_type="dataset"), encoding="utf8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row["id"])) in banned:
                    dropped += 1
                    continue
            except (KeyError, ValueError):
                continue
            en, ar = (row.get("caption_en") or "").strip(), (row.get("caption_ar") or "").strip()
            if en and ar:
                pairs.append((en, ar))
            if limit and len(pairs) >= limit:
                break
    print(f"Arabic MS-COCO: {len(pairs)} caption pairs kept, {dropped} dropped because their photo is in XTD")
    return pairs
