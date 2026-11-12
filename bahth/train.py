"""Fine-tune a multilingual text encoder for image search in Arabic, Turkish and English.

    python -m bahth.train --data data

The image tower (CLIP ViT-B/32) stays frozen, so image embeddings are computed once and
training only moves the text encoder. Three recipes are run from the same starting point:

  contrastive   pull each Flickr8k caption (ar, tr, en) towards its own photo and away
                from the other photos in the batch (symmetric InfoNCE)
  distill       no photos at all: make the Arabic caption land where CLIP's English text
                encoder puts its English translation (Arabic MS-COCO pairs)
  both          the two objectives in alternation

Every recipe is scored on Flickr8k test (same kind of photos as training) and on XTD
(MS-COCO photos, never trained on), next to the untouched starting model.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluate import recall_at_k

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LANGS = ("en", "ar", "tr")


def embed(student, texts: list[str]) -> torch.Tensor:
    """Text embeddings with gradients, L2-normalised."""
    feats = {k: v.to(DEV) for k, v in student.tokenize(texts).items() if hasattr(v, "to")}
    return F.normalize(student(feats)["sentence_embedding"].float(), dim=-1)


@torch.no_grad()
def embed_eval(student, texts: list[str], batch: int = 512) -> torch.Tensor:
    was = student.training
    student.eval()
    out = torch.cat([embed(student, texts[i:i + batch]).cpu() for i in range(0, len(texts), batch)])
    student.train(was)
    return out


def score(student, img_emb: torch.Tensor, index: dict[str, int], caps: dict, names: list[str]) -> dict:
    """Recall for each language over the gallery formed by `names`."""
    rows = [index[n] for n in names]
    gallery = img_emb[rows]
    out = {}
    for lang in LANGS:
        q, t = [], []
        for k, n in enumerate(names):
            for c in caps.get(lang, {}).get(n, []):
                q.append(c)
                t.append(k)
        if q:
            out[lang] = recall_at_k(embed_eval(student, q), gallery, t)
    return out


def info_nce(text: torch.Tensor, image: torch.Tensor, scale: float) -> torch.Tensor:
    logits = scale * text @ image.T
    target = torch.arange(len(text), device=text.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def run(student, teacher_text, flickr, coco_pairs, recipe: str, steps: int, batch: int = 128, lr: float = 2e-5,
        scale: float = 100.0, eval_every: int = 100, seed: int = 0, log=print) -> tuple[dict, dict]:
    """Trains one recipe. Returns (best state dict, history). `flickr` = (img_emb, index, caps, train_names, dev_names)."""
    img_emb, index, caps, train_names, dev_names = flickr
    rng = random.Random(seed)
    torch.manual_seed(seed)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    use_c, use_d = recipe in ("contrastive", "both"), recipe in ("distill", "both")
    best, best_state, history = -1.0, None, []
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = torch.zeros((), device=DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            if use_c:
                names = rng.sample(train_names, batch)          # distinct photos, so no false negatives in the batch
                image = img_emb[[index[n] for n in names]].to(DEV)
                for lang in LANGS:
                    keep = [k for k, n in enumerate(names) if caps[lang].get(n)]
                    texts = [rng.choice(caps[lang][names[k]]) for k in keep]
                    loss = loss + info_nce(embed(student, texts), image[keep], scale) / len(LANGS)
            if use_d:
                pairs = rng.sample(coco_pairs, batch)
                target = teacher_text([en for en, _ in pairs]).to(DEV)
                both = embed(student, [ar for _, ar in pairs] + [en for en, _ in pairs])   # keep English in place too
                loss = loss + (1 - (both * torch.cat([target, target])).sum(-1)).mean() * 10
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % eval_every == 0 or step == steps:
            m = score(student, img_emb, index, caps, dev_names)
            sel = sum(m[l]["R@1"] for l in ("ar", "tr") if l in m) / 2
            history.append({"step": step, "loss": loss.item(), **{f"dev_{l}_R@1": m[l]["R@1"] for l in m}})
            flag = ""
            if sel > best:
                best, best_state, flag = sel, copy.deepcopy({k: v.detach().cpu() for k, v in student.state_dict().items()}), " *"
            log(f"[{recipe}] step {step:4d}  loss {loss.item():.3f}  dev R@1 " + "  ".join(f"{l} {m[l]['R@1']:.3f}" for l in m) + f"  {time.time() - t0:.0f}s{flag}")
    return best_state, {"recipe": recipe, "steps": steps, "best_dev_mean_R@1_ar_tr": best, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--student", default="sentence-transformers/clip-ViT-B-32-multilingual-v1")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    from PIL import Image
    from sentence_transformers import SentenceTransformer

    from .data import captions, image_path, prepare, splits
    from .ood import coco_parallel, prepare_xtd

    root = prepare(a.data)
    sp, caps = splits(root), captions(root)
    clip = SentenceTransformer("clip-ViT-B-32", device=DEV)

    def images(paths):
        return clip.encode([Image.open(p).convert("RGB") for p in paths], batch_size=128, convert_to_tensor=True, normalize_embeddings=True).float().cpu()

    @torch.no_grad()
    def teacher_text(texts):
        return clip.encode(texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True).float()

    names = sp["train"] + sp["dev"] + sp["test"]
    index = {n: i for i, n in enumerate(names)}
    img_emb = images([image_path(root, n) for n in names])
    flickr = (img_emb, index, caps, sp["train"], sp["dev"])
    print(f"Flickr8k: {len(names)} photos embedded")

    xtd, coco = None, []
    try:
        xnames, xcaps, xdir = prepare_xtd(root)
        xtd = (images([xdir / n for n in xnames]), {n: i for i, n in enumerate(xnames)}, xcaps, xnames)
        coco = coco_parallel(xnames)
    except Exception as e:      # the in-domain experiment still runs if a download fails
        print(f"out-of-domain data unavailable: {type(e).__name__}: {e}")

    def evaluate(student):
        r = {"flickr8k_test": score(student, img_emb, index, caps, sp["test"])}
        if xtd:
            r["xtd"] = score(student, xtd[0], xtd[1], xtd[2], xtd[3])
        return r

    def fresh():
        s = SentenceTransformer(a.student, device=DEV)
        s.max_seq_length = 64
        return s

    results = {"student": a.student, "parameters": sum(p.numel() for p in fresh().parameters()), "runs": {}}
    results["runs"]["starting model"] = {"metrics": evaluate(fresh())}
    best_name, best_sel = None, -1.0
    for recipe in ("contrastive", "distill", "both"):
        if recipe != "contrastive" and not coco:
            continue
        student = fresh()
        state, hist = run(student, teacher_text, flickr, coco, recipe, a.steps, a.batch, a.lr)
        student.load_state_dict(state)
        results["runs"][recipe] = {"metrics": evaluate(student), **hist}
        if hist["best_dev_mean_R@1_ar_tr"] > best_sel:
            best_name, best_sel = recipe, hist["best_dev_mean_R@1_ar_tr"]
            student.save(str(out / "student"))
        (out / "finetune.json").write_text(json.dumps(results, indent=1))
        del student
        torch.cuda.empty_cache()
    results["selected_on_dev"] = best_name
    (out / "finetune.json").write_text(json.dumps(results, indent=1))

    print("\nRecall@1 / @5 / @10, text -> image (percent)")
    for name, r in results["runs"].items():
        for sset, m in r["metrics"].items():
            print(f"{name:15s} {sset:14s} " + "   ".join(f"{l} {m[l]['R@1']*100:4.1f}/{m[l]['R@5']*100:4.1f}/{m[l]['R@10']*100:4.1f}" for l in m))
    print("selected on dev:", best_name)


if __name__ == "__main__":
    main()
