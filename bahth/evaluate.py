"""Text-to-image retrieval metrics.

Every caption is a query; the gallery is the images of one split. Recall@K is the
share of queries whose own image is among the K most similar images.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def recall_at_k(text_emb: torch.Tensor, image_emb: torch.Tensor, target: list[int], ks=(1, 5, 10)) -> dict:
    """text_emb [Q,D], image_emb [N,D], both L2-normalised; target[q] = index of the right image."""
    t = torch.as_tensor(target, device=text_emb.device)
    ranks = []
    for i in range(0, len(text_emb), 2048):
        sim = text_emb[i:i + 2048] @ image_emb.T
        own = sim.gather(1, t[i:i + 2048, None])
        ranks.append((sim > own).sum(1))          # number of images scored above the right one
    ranks = torch.cat(ranks).cpu().numpy()
    out = {f"R@{k}": float((ranks < k).mean()) for k in ks}
    out.update(median_rank=float(np.median(ranks) + 1), queries=int(len(ranks)), gallery=int(image_emb.shape[0]))
    return out


def evaluate_model(encode_images, encode_texts, images: list, caps: dict[str, dict[str, list[str]]], names: list[str]) -> dict:
    """Scores one model on every language. `encode_*` return L2-normalised float tensors."""
    img = encode_images(images)
    out = {}
    for lang, by_file in caps.items():
        queries, target = [], []
        for idx, n in enumerate(names):
            for c in by_file.get(n, []):
                queries.append(c)
                target.append(idx)
        out[lang] = recall_at_k(encode_texts(queries), img, target)
    return out
