"""Score the pretrained baselines on the Flickr8k test images in English, Arabic and Turkish.

    python -m bahth.baselines --data data

Writes results/baselines.json. These are the numbers a fine-tuned model has to beat.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image

from .data import captions, image_path, prepare, splits
from .evaluate import evaluate_model
from .models import BASELINES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    root = prepare(a.data)
    names = splits(root)[a.split]
    caps = captions(root)
    images = [Image.open(image_path(root, n)).convert("RGB") for n in names]
    print(f"{a.split}: {len(names)} images; queries per language: " + ", ".join(f"{l} {sum(len(caps[l].get(n, [])) for n in names)}" for l in caps))

    results = {}
    for name, make in BASELINES.items():
        t0 = time.time()
        try:
            enc_i, enc_t = make()
            results[name] = evaluate_model(enc_i, enc_t, images, caps, names)
            row = "  ".join(f"{l}: R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f}" for l, m in results[name].items())
            print(f"{name}\n   {row}   ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # one unavailable checkpoint should not lose the others
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{name}\n   FAILED: {results[name]['error']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, "baselines.json").write_text(json.dumps({"split": a.split, "images": len(names), "models": results}, indent=1))


if __name__ == "__main__":
    main()
