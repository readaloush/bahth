"""Serving path with a tiny stand-in model: tokenizer -> ONNX -> index -> HTTP."""
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from bahth.search import Index, TextEncoder

WORDS = ["[PAD]", "[UNK]", "dog", "cat", "beach", "كلب", "قطة", "köpek", "kedi", "a", "on", "the"]
ANIMAL = {"dog": 0, "كلب": 0, "köpek": 0, "cat": 1, "قطة": 1, "kedi": 1}


class Tiny(torch.nn.Module):
    """Embedding table where translations of a word share a direction."""

    def __init__(self):
        super().__init__()
        w = torch.zeros(len(WORDS), 4)
        for word, axis in ANIMAL.items():
            w[WORDS.index(word), axis] = 1.0
        w[WORDS.index("beach"), 2] = 1.0
        self.emb = torch.nn.Embedding.from_pretrained(w)

    def forward(self, input_ids, attention_mask):
        m = attention_mask.unsqueeze(-1).float()
        return (self.emb(input_ids) * m).sum(1) / m.sum(1).clamp(min=1)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(d / "tokenizer.json"))
    ids = torch.tensor([[2, 4, 0]])
    torch.onnx.export(Tiny(), (ids, (ids > 0).long()), str(d / "text.int8.onnx"), dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "embedding": {0: "b"}})
    gallery = np.eye(4, dtype=np.float16)[:3]
    np.save(d / "gallery.npy", gallery)
    photos = [{"id": n, "image": f"https://example.test/{n}", "by": "x", "user": "x", "w": 4, "h": 3, "blur": ""} for n in ("dog", "cat", "beach")]
    (d / "gallery.json").write_text(json.dumps(photos))
    return d


def test_encoder_is_normalised_and_ignores_padding(artifacts):
    enc = TextEncoder(artifacts / "text.int8.onnx", artifacts / "tokenizer.json")
    v = enc(["dog", "a dog on the beach", "cat"])
    assert v.shape == (3, 4)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert np.allclose(enc(["dog"])[0], v[0], atol=1e-6)      # same vector alone or padded in a batch


def test_same_photo_in_three_languages(artifacts):
    enc = TextEncoder(artifacts / "text.int8.onnx", artifacts / "tokenizer.json")
    index = Index.load(artifacts)
    for q, want in [("dog", "dog"), ("كلب", "dog"), ("köpek", "dog"), ("قطة", "cat"), ("kedi", "cat")]:
        assert index.search(enc([q])[0], 1)[0]["id"] == want


def test_search_orders_by_score_and_caps_k(artifacts):
    index = Index.load(artifacts)
    hits = index.search(np.array([0.2, 0.9, 0.5, 0], dtype=np.float32), 50)
    assert [h["id"] for h in hits] == ["cat", "beach", "dog"]
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_index_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        Index(np.zeros((2, 4)), [{"id": "a"}])


def test_http(artifacts, monkeypatch):
    from fastapi.testclient import TestClient

    import service.app as svc
    monkeypatch.setattr(svc, "ARTIFACTS", artifacts)
    svc.state.cache_clear()
    client = TestClient(svc.app)
    assert client.get("/health").json() == {"status": "ok", "photos": 3, "model": "text.int8.onnx"}
    r = client.get("/search", params={"q": "قطة", "k": 2}).json()
    assert r["photos"][0]["id"] == "cat" and len(r["photos"]) == 2
    assert r["photos"][0]["page"] == "https://unsplash.com/photos/cat"
    assert client.get("/search", params={"q": "   "}).status_code == 422
    assert client.get("/search", params={"q": "dog", "k": 500}).status_code == 422
    svc.state.cache_clear()


def test_http_reports_missing_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import service.app as svc
    monkeypatch.setattr(svc, "ARTIFACTS", tmp_path)
    svc.state.cache_clear()
    r = TestClient(svc.app).get("/health")
    assert r.status_code == 503 and "gallery.npy" in r.json()["detail"]
    svc.state.cache_clear()
