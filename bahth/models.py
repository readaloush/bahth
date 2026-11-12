"""Uniform wrappers around the pretrained image-text models used as baselines.

Each factory returns (encode_images, encode_texts); both take a list and return an
L2-normalised float tensor on the CPU, so models can be compared with the same code.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def open_clip_model(arch: str, pretrained: str):
    """Any OpenCLIP checkpoint, e.g. ('ViT-B-32', 'openai') or ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k')."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=DEVICE)
    tokenizer = open_clip.get_tokenizer(arch)
    model.eval()

    @torch.no_grad()
    def encode_images(images):
        out = [model.encode_image(torch.stack([preprocess(im.convert("RGB")) for im in b]).to(DEVICE)) for b in _batched(images, 128)]
        return F.normalize(torch.cat(out).float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(texts):
        out = [model.encode_text(tokenizer(b).to(DEVICE)) for b in _batched(texts, 256)]
        return F.normalize(torch.cat(out).float(), dim=-1).cpu()

    return encode_images, encode_texts


def sentence_transformers_pair(image_model: str = "clip-ViT-B-32", text_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"):
    """The classic multilingual CLIP: a multilingual text encoder distilled into CLIP ViT-B/32's space."""
    from sentence_transformers import SentenceTransformer

    img_m, txt_m = SentenceTransformer(image_model, device=DEVICE), SentenceTransformer(text_model, device=DEVICE)

    def encode_images(images):
        return img_m.encode([im.convert("RGB") for im in images], batch_size=128, convert_to_tensor=True, normalize_embeddings=True).float().cpu()

    def encode_texts(texts):
        return txt_m.encode(list(texts), batch_size=256, convert_to_tensor=True, normalize_embeddings=True).float().cpu()

    return encode_images, encode_texts


BASELINES = {
    "CLIP ViT-B/32 (OpenAI, English only)": lambda: open_clip_model("ViT-B-32", "openai"),
    "Multilingual CLIP ViT-B/32 (sentence-transformers)": lambda: sentence_transformers_pair(),
    "XLM-R base + ViT-B/32 (LAION-5B)": lambda: open_clip_model("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
}
