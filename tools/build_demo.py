"""Build the static demo assets in docs/ from artifacts/.

    python tools/build_demo.py

- docs/model/text.partNN      the int8 text encoder with its vocabulary cut to Latin- and
                              Arabic-script tokens (the three demo languages), in 3.5 MB parts
- docs/model/vocab.json       the kept tokens, in their new id order
- docs/gallery/emb.partNN     image embeddings as little-endian float16
- docs/gallery/photos.json    photo id, image path, photographer, aspect ratio
- docs/examples.json          ready-made results for the example queries
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bahth.search import Index, TextEncoder  # noqa: E402

ART, DOCS, PART = Path("artifacts"), Path("docs"), 3_500_000
EMB = "student.0.model.embeddings.word_embeddings.weight_quantized"

EXAMPLES = {
    "en": ["a lighthouse at sunset", "a cat sleeping on a sofa", "a crowded market at night", "people hiking in snowy mountains",
           "a cup of coffee on a wooden table", "a rainy city street", "children playing football", "an old door painted blue"],
    "ar": ["منارة عند غروب الشمس", "قطة نائمة على أريكة", "سوق مزدحم في الليل", "أشخاص يمشون في جبال ثلجية",
           "فنجان قهوة على طاولة خشبية", "شارع في مدينة ممطرة", "أطفال يلعبون كرة القدم", "باب قديم مطلي باللون الأزرق"],
    "tr": ["gün batımında deniz feneri", "kanepede uyuyan kedi", "gece kalabalık bir pazar", "karlı dağlarda yürüyen insanlar",
           "ahşap masada bir fincan kahve", "yağmurlu bir şehir sokağı", "futbol oynayan çocuklar", "maviye boyanmış eski bir kapı"],
}


def keep_token(tok: str) -> bool:
    if tok.startswith("[") and tok.endswith("]"):
        return True
    for ch in tok[2:] if tok.startswith("##") else tok:
        o = ord(ch)
        latin = o < 0x250 or 0x300 <= o <= 0x36F or 0x2000 <= o <= 0x206F or 0x20A0 <= o <= 0x20CF
        arabic = 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF
        if not (latin or arabic):
            return False
    return True


def parts(data: bytes, folder: Path, stem: str) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for k, i in enumerate(range(0, len(data), PART)):
        names.append(f"{stem}.part{k:02d}")
        (folder / names[-1]).write_bytes(data[i:i + PART])
    return names


def main():
    for d in ("model", "gallery"):
        shutil.rmtree(DOCS / d, ignore_errors=True)
    tok = json.loads((ART / "tokenizer.json").read_text())
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda kv: kv[1])
    kept = [(t, i) for t, i in vocab if keep_token(t)]
    rows = np.array([i for _, i in kept])

    model = onnx.load(str(ART / "text.int8.onnx"))
    init = next(i for i in model.graph.initializer if i.name == EMB)
    table = numpy_helper.to_array(init)
    init.CopyFrom(numpy_helper.from_array(table[rows], EMB))
    blob = model.SerializeToString()
    (ART / "text.web.onnx").write_bytes(blob)
    model_parts = parts(blob, DOCS / "model", "text")
    (DOCS / "model" / "vocab.json").write_text(json.dumps([t for t, _ in kept], ensure_ascii=False, separators=(",", ":")))

    index = Index.load(ART)
    # float16, not int8: a few CLIP dimensions are ten times larger than the rest and int8 reorders the top results
    f16 = index.emb.astype("<f2")
    emb_parts = parts(f16.tobytes(), DOCS / "gallery", "emb")
    prefix = "https://images.unsplash.com/"
    assert all(p["image"].startswith(prefix) for p in index.photos)
    compact = [[p["id"], p["image"][len(prefix):], p["by"], p["user"], round(p["w"] / p["h"], 3) if p["h"] else 1.5] for p in index.photos]
    (DOCS / "gallery" / "photos.json").write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))

    enc = TextEncoder(ART / "text.int8.onnx", ART / "tokenizer.json")
    examples = {}
    for lang, qs in EXAMPLES.items():
        examples[lang] = []
        for q in qs:
            v = enc([q])[0]
            hits = index.search(v, 12)
            examples[lang].append({"q": q, "hits": [[h["id"], h["image"][len(prefix):], h["by"], h["user"],
                                                      round(h["w"] / h["h"], 3) if h["h"] else 1.5, h["score"]] for h in hits]})
    (DOCS / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, separators=(",", ":")))
    manifest = {"model": model_parts, "model_bytes": len(blob), "vocab": len(kept), "full_vocab": len(vocab),
                "gallery": emb_parts, "photos": len(index), "dim": int(f16.shape[1]), "max_tokens": 64}
    (DOCS / "manifest.json").write_text(json.dumps(manifest))
    print(manifest["vocab"], "of", manifest["full_vocab"], "tokens kept;", round(len(blob) / 1e6, 1), "MB in", len(model_parts), "parts;",
          len(emb_parts), "gallery parts")


if __name__ == "__main__":
    main()
