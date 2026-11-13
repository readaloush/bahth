# Bahth بحث

Search photos by typing in Arabic, Turkish or English.

A compact multilingual CLIP finds the right photo first for about half of English queries but only a quarter of Arabic and Turkish ones. This project measures that gap, closes most of it by fine-tuning the text encoder, and checks the gain on photos the model never trained on.

**Live demo: https://readaloush.github.io/bahth/** The example searches open at once; typing your own query downloads the model (99 MB) and runs it in the browser.

## Results

Text-to-image retrieval, Recall@1 / @5 / @10 in percent. One training run per recipe, 1,200 steps each on a Kaggle T4, no repeated seeds, so differences of a point or so are within noise.

Flickr8k test, 1,000 photos (queries: 5,000 English, 3,000 Arabic, 2,000 Turkish). Same kind of photos as the training data.

| Model | English | Arabic | Turkish |
|---|---|---|---|
| OpenAI CLIP ViT-B/32 | 53.4 / 79.2 / 88.1 | 0.1 / 0.8 / 1.4 | 0.9 / 3.2 / 5.5 |
| XLM-R + ViT-B/32 (LAION-5B), not fine-tuned | 59.0 / 84.6 / 91.2 | 38.5 / 64.5 / 75.4 | 42.6 / 70.2 / 81.2 |
| Multilingual CLIP (starting model) | 48.9 / 76.8 / 85.8 | 24.8 / 50.1 / 61.0 | 25.8 / 52.2 / 62.6 |
| + contrastive | 61.4 / 87.7 / 93.7 | 47.6 / 78.4 / 87.4 | 49.5 / 81.0 / 88.7 |
| + distillation | 50.0 / 77.9 / 86.7 | 32.4 / 60.9 / 72.5 | 26.8 / 54.0 / 65.8 |
| + both (selected on dev) | 60.6 / 86.4 / 92.9 | 46.5 / 77.1 / 87.0 | 49.5 / 80.2 / 88.6 |

XTD, 1,000 MS-COCO photos with one caption per language. No XTD photo is in the training data.

| Model | English | Arabic | Turkish |
|---|---|---|---|
| Multilingual CLIP (starting model) | 47.3 / 77.2 / 88.3 | 23.2 / 43.1 / 53.1 | 33.3 / 61.1 / 74.4 |
| + contrastive | 55.7 / 83.7 / 92.0 | 32.3 / 53.5 / 63.2 | 42.5 / 72.6 / 84.7 |
| + distillation | 50.7 / 79.1 / 89.7 | 34.6 / 59.8 / 69.7 | 36.4 / 65.0 / 77.0 |
| + both (selected on dev) | 51.3 / 80.3 / 90.9 | 35.4 / 59.0 / 69.5 | 44.6 / 73.1 / 84.1 |

What the numbers say:

- Arabic and Turkish Recall@1 on Flickr8k roughly doubles (24.8 to 46.5, 25.8 to 49.5). On unseen XTD photos the gain is about half as large (23.2 to 35.4, 33.3 to 44.6). The in-domain figure overstates what a user would see on other photos.
- Contrastive training alone wins in-domain but transfers worst to Arabic on XTD. Distillation from the English teacher on MS-COCO caption pairs transfers best for Arabic and does almost nothing for Turkish, which has no pairs in that data.
- Combining them costs English on XTD (55.7 down to 51.3) compared with contrastive alone. That trade-off is open.
- The fine-tuned model beats the larger XLM-R model on Flickr8k, but that model was not scored on XTD here, so this is an in-domain comparison only.

## How it works

The image tower (CLIP ViT-B/32) stays frozen, so photo embeddings never change and an index built once stays valid. Only the multilingual text encoder (135 M parameters) is trained, three ways from the same starting point:

- contrastive: InfoNCE between captions and photos, with Arabic, Turkish and English captions in each batch
- distillation: pull the Arabic and English embeddings of a caption pair toward the English CLIP text embedding
- both

The checkpoint is chosen by mean Arabic and Turkish Recall@1 on the Flickr8k dev split. Test and XTD are scored once at the end.

## Serving

The service answers a query in two steps: the text encoder turns it into a vector (ONNX Runtime, no PyTorch in the image), then an exact cosine search ranks 24,997 Unsplash photos.

Exporting has a cost, measured on the same queries through the same code the service runs. Recall@1 / @5 / @10 in percent:

| Text encoder | Size | CPU per query | Flickr8k Arabic | Flickr8k Turkish | XTD Arabic | XTD Turkish |
|---|---|---|---|---|---|---|
| PyTorch | | | 46.5 / 77.1 / 87.0 | 49.5 / 80.2 / 88.6 | 35.4 / 59.0 / 69.5 | 44.6 / 73.1 / 84.1 |
| ONNX float32 | 541 MB | 10.8 ms | 46.5 / 77.1 / 87.0 | 49.5 / 80.2 / 88.6 | 35.4 / 59.0 / 69.5 | 44.6 / 73.1 / 84.1 |
| ONNX int8 | 136 MB | 5.0 ms | 44.5 / 74.8 / 85.7 | 47.3 / 78.2 / 87.8 | 34.2 / 58.4 / 68.6 | 41.1 / 70.2 / 83.5 |

The float32 export is exact. Int8 quantisation makes the model four times smaller and about twice as fast, and costs 1 to 3.5 points of Recall@1 in Arabic and Turkish (0.4 to 0.8 in English). The service uses int8 by default; set `BAHTH_MODEL=text.onnx` to trade size for accuracy. Timings are for one short query on a Kaggle CPU.

The index is a matrix product over 24,997 vectors, 2 to 4 ms per query. At this size an approximate index would add a dependency and lose recall for nothing.

The gallery is the Unsplash Lite dataset. Its terms do not allow redistributing the photos, so only embeddings, photo ids and photographer credits are stored, and results link to Unsplash. 3 of 25,000 photos failed to download and are left out.

```
# build text.int8.onnx, tokenizer.json, gallery.npy, gallery.json into artifacts/
#   notebooks/03_index.ipynb on Kaggle, or:
python -m bahth.export --student results/student --data data --out artifacts
python -m bahth.gallery --out artifacts

BAHTH_ARTIFACTS=artifacts uvicorn service.app:app
curl 'localhost:8000/search?q=kanepede uyuyan kedi&k=5'

docker build -t bahth .
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" bahth
```

The tests (`pytest tests`) cover the serving path with a small stand-in model. The Docker image has not been built in CI yet.

The demo page (`docs/`) runs the same int8 model with ONNX Runtime Web. `tools/build_demo.py` cuts the vocabulary to Latin- and Arabic-script tokens (119,547 to 72,513, which saves 36 MB and changes nothing for queries in those scripts) and writes the files in parts. `tests/browser_check.py` drives the page in headless Chrome and checks that its tokeniser returns the same ids as the `tokenizers` library and that typed queries return the same photos as the Python service.

## Run it

The notebooks run on Kaggle with a GPU and internet on:

- [`notebooks/01_baselines.ipynb`](notebooks/01_baselines.ipynb) scores the three existing models
- [`notebooks/02_finetune.ipynb`](notebooks/02_finetune.ipynb) trains the three recipes (about 39 minutes on a T4) and writes `results/finetune.json`
- [`notebooks/03_index.ipynb`](notebooks/03_index.ipynb) exports the model, embeds the gallery and runs sample queries

```
pip install -r requirements.txt
python -m bahth.baselines --data data
python -m bahth.train --data data --steps 1200
```

## Data

None of it is redistributed here; the code downloads it.

- Flickr8k (Hodosh, Young & Hockenmaier, 2013), split 6,000 / 1,000 / 1,000
- Arabic Flickr8k captions (ElJundi et al., 2020)
- TasvirEt Turkish captions (Unal et al., 2016), matched to Flickr8k photos by thumbnail
- XTD10 (Aggarwal & Kale, 2020) with the Arabic captions from XTD-11
- Arabic MS-COCO caption pairs for distillation, with every XTD photo removed first
- Unsplash Lite (unsplash.com/data) as the search gallery

## Licence

Code under the licence in [`LICENSE`](LICENSE). Model weights and datasets keep their own licences.
