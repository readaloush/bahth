"""Export the fine-tuned text encoder for serving and measure what the export costs.

    python -m bahth.export --student results/student --data data --out artifacts

Writes text.onnx (float32), text.int8.onnx (dynamic int8 quantisation) and tokenizer.json,
then scores PyTorch, ONNX float32 and ONNX int8 on the same queries through the same
code the service runs (bahth.search.TextEncoder). Results go to export.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import recall_at_k

LANGS = ("en", "ar", "tr")


class Wrapper(torch.nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, input_ids, attention_mask):
        out = self.student({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"]
        return torch.nn.functional.normalize(out, dim=-1)


def export(student, out: Path) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    student = student.cpu().eval()
    student.tokenizer.save_pretrained(str(out / "tokenizer"))
    (out / "tokenizer.json").write_bytes((out / "tokenizer" / "tokenizer.json").read_bytes())
    feats = student.tokenize(["a dog runs on the beach", "كلب"])
    args = (feats["input_ids"], feats["attention_mask"])
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["embedding"], opset_version=17,
              dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "embedding": {0: "batch"}})
    with torch.no_grad():
        try:
            torch.onnx.export(Wrapper(student), args, str(out / "text.onnx"), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(Wrapper(student), args, str(out / "text.onnx"), **kw)
    quantize_dynamic(str(out / "text.onnx"), str(out / "text.int8.onnx"), weight_type=QuantType.QInt8)
    return {f: round((out / f).stat().st_size / 1e6, 1) for f in ("text.onnx", "text.int8.onnx")}


def queries(caps: dict, names: list[str]) -> dict[str, tuple[list[str], list[int]]]:
    out = {}
    for lang in LANGS:
        q, t = [], []
        for k, n in enumerate(names):
            for c in caps.get(lang, {}).get(n, []):
                q.append(c)
                t.append(k)
        if q:
            out[lang] = (q, t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="results/student")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    out = Path(a.out)

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    from .data import captions, image_path, prepare, splits
    from .ood import prepare_xtd
    from .search import TextEncoder
    from .train import embed_eval

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    student = SentenceTransformer(a.student, device=dev)
    student.max_seq_length = 64
    clip = SentenceTransformer("clip-ViT-B-32", device=dev)

    def images(paths):
        return clip.encode([Image.open(p).convert("RGB") for p in paths], batch_size=128, convert_to_tensor=True, normalize_embeddings=True).float().cpu()

    root = prepare(a.data)
    test = splits(root)["test"]
    sets = {"flickr8k_test": (images([image_path(root, n) for n in test]), queries(captions(root), test))}
    try:
        xnames, xcaps, xdir = prepare_xtd(root)
        sets["xtd"] = (images([xdir / n for n in xnames]), queries(xcaps, xnames))
    except Exception as e:
        print(f"XTD unavailable: {type(e).__name__}: {e}")

    report = {"metrics": {}, "max_abs_difference": {}}
    torch_emb = {s: {l: embed_eval(student, q) for l, (q, _) in qs.items()} for s, (_, qs) in sets.items()}
    report["metrics"]["pytorch"] = {s: {l: recall_at_k(torch_emb[s][l], g, qs[l][1]) for l in qs} for s, (g, qs) in sets.items()}

    report["size_mb"] = export(student, out)
    for name, f in (("onnx float32", "text.onnx"), ("onnx int8", "text.int8.onnx")):
        enc = TextEncoder(out / f, out / "tokenizer.json")
        t0 = time.time()
        enc(["a dog runs on the beach"] * 32, batch=1)
        ms = (time.time() - t0) / 32 * 1000
        m, diff = {}, 0.0
        for s, (g, qs) in sets.items():
            m[s] = {}
            for l, (q, t) in qs.items():
                e = torch.from_numpy(enc(q))
                diff = max(diff, float((e - torch_emb[s][l]).abs().max()))
                m[s][l] = recall_at_k(e, g, t)
        report["metrics"][name] = m
        report["max_abs_difference"][name] = diff
        report.setdefault("cpu_ms_per_query", {})[name] = round(ms, 1)
    (out / "export.json").write_text(json.dumps(report, indent=1))

    print("sizes (MB):", report["size_mb"], "| CPU ms per query:", report["cpu_ms_per_query"])
    print("largest embedding difference from PyTorch:", {k: round(v, 4) for k, v in report["max_abs_difference"].items()})
    print("Recall@1 / @5 / @10 (percent)")
    for name, r in report["metrics"].items():
        for s, m in r.items():
            print(f"{name:13s} {s:14s} " + "   ".join(f"{l} {m[l]['R@1']*100:4.1f}/{m[l]['R@5']*100:4.1f}/{m[l]['R@10']*100:4.1f}" for l in m))


if __name__ == "__main__":
    main()
