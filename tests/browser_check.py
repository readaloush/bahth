"""Check the demo page in headless Chrome against the Python serving path.

    python tests/browser_check.py            (needs artifacts/ and a local copy of onnxruntime-web in --ort)

Tokeniser ids must equal the tokenizers library's, and typed queries must return the same
top photos as bahth.search.
"""
import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = ["a lighthouse at sunset", "منارة عند غروب الشمس", "gün batımında deniz feneri", "İstanbul'da çay içen yaşlı adam!",
           "قطّة صغيرة، نائمة؟", "two dogs (running) on the beach...", "ÇIĞLIK atan çocuk; ışık ve gölge", "café naïve",
           "طفلٌ يَلعبُ بالكرة", "   spaces\tand\nnewlines  ", "x" * 120, "price: $20 + 5% = ?", "a " * 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ort", default=str(ROOT.parent / "lawha" / "docs" / "ort"))
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    from tokenizers import Tokenizer

    from bahth.search import Index, TextEncoder

    docs = ROOT / "docs"
    link = docs / "ort"
    if not link.exists():
        link.symlink_to(a.ort)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(docs))
    handler.log_message = lambda *x, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/?ort=ort/&lang=en"

    tok = Tokenizer.from_file(str(ROOT / "artifacts" / "tokenizer.json"))
    tok.enable_truncation(max_length=64)
    vocab = json.loads((docs / "model" / "vocab.json").read_text())
    new = {t: i for i, t in enumerate(vocab)}
    enc = TextEncoder(ROOT / "artifacts" / "text.int8.onnx", ROOT / "artifacts" / "tokenizer.json")
    index = Index.load(ROOT / "artifacts")
    bad = 0
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome")
            page = b.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(url)
            page.wait_for_selector("#frames li")
            assert page.locator("#frames li").count() == 12, "example result did not render"
            page.fill("#q", QUERIES[0] + " ")
            page.fill("#q", "a red bicycle")
            page.click("button.go")
            page.wait_for_selector("#notice:not([hidden])")
            page.click("#load")
            page.wait_for_function("window.bahth", timeout=240000)
            page.wait_for_selector("#notice", state="hidden", timeout=60000)
            for q in QUERIES:
                want = [new.get(t, new["[UNK]"]) for t in tok.encode(q).tokens]
                got = page.evaluate("q => window.bahth.ids(q)", q)
                if got != want:
                    bad += 1
                    print("TOKENS DIFFER", repr(q[:40]), got[:12], want[:12])
            for q in QUERIES[:9]:
                ref = [h["id"] for h in index.search(enc([q])[0], 12)]
                got = page.evaluate("async q => (window.bahth.search(await window.bahth.encode(q), 12)).map(h => h[0])", q)
                same = len(set(ref) & set(got))
                if got[0] != ref[0] or same < 9:      # int8 kernels differ slightly between wasm and native; near-ties may swap
                    bad += 1
                print(f"{same:2d}/12 same, first {'same' if got[0] == ref[0] else 'DIFFERENT'}  {q[:40]}")
            print("edge print:", page.inner_text("#edgeR"), "| page errors:", errors)
            page.screenshot(path="/tmp/bt/desktop.png", full_page=True)
            for lang, w in (("ar", 1280), ("tr", 390)):
                pg = b.new_page(viewport={"width": w, "height": 844})
                pg.goto(url.replace("lang=en", f"lang={lang}"))
                pg.wait_for_selector("#frames li")
                pg.screenshot(path=f"/tmp/bt/{lang}_{w}.png", full_page=True)
            b.close()
    finally:
        srv.shutdown()
        link.unlink()
    print("FAILED" if bad else "OK", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
