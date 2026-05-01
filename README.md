# 🦋 Butterfly Species Identifier

Upload a field photo → get the **species name**, **IUCN conservation status**, and a **Wikipedia summary** — instantly, in your browser.

Built for SMAI Assignment T7.4 using the [100-species Kaggle dataset](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

---

## Results at a Glance

| Model | Strategy | Test Top-1 | Test Top-3 | Macro F1 |
|---|---|:---:|:---:|:---:|
| Author Keras H5 (baseline) | Pre-trained, no extra training | 97.60% | 99.60% | — |
| **EfficientNet-B0 ✅** | Linear probe → fine-tune last block | **96.60%** | **99.40%** | **96.45%** |
| CLIP ViT-B/32 | Zero-shot, 5-prompt ensemble | 28.20% | 44.20% | 24.57% |

The deployed app uses the **EfficientNet-B0** model (16 MB `.pth` file).

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Anurag-Kacholiya/Butterfly-Species-Detection.git
cd Butterfly-Species-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ required. Use a virtual environment:
> ```bash
> python -m venv .venv && source .venv/bin/activate   # macOS / Linux
> python -m venv .venv && .venv\Scripts\activate       # Windows
> pip install -r requirements.txt
> ```

### 3. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** — drop a butterfly photo and see predictions.

---

## Project Structure

```
Butterfly-Species-Detection/
│
├── app.py                  # Streamlit app (entry point)
├── requirements.txt        # pip dependencies
├── report.tex              # LaTeX technical report
│
├── model/
│   └── efficientnet_butterfly.pth   # Fine-tuned EfficientNet-B0 weights (16 MB)
│
├── data/
│   ├── class_names.json             # Ordered list of 100 species
│   ├── species_metadata.json        # Wiki summaries, IUCN status, fun facts
│   └── species_images/              # Reference photos shown in the UI
│       └── *.jpg                    # One image per species
│
├── notebooks/
│   ├── 01_efficientnet_finetuning.ipynb   # EfficientNet training pipeline (run on Kaggle)
│   └── 02_clip_zeroshot.ipynb            # CLIP zero-shot pipeline (run on Kaggle)
│
└── results/
    ├── efficientnet/        # Training curves, confusion matrix, Grad-CAM, UMAP, ablation
    └── clip/                # Ablation plots, confusion matrix, UMAP, misclassifications
```

> The raw dataset (`archive/`) is not committed — download from Kaggle:
> `kaggle datasets download gpiosenka/butterfly-images40-species`

---

## Methodology

### Approach 1 — Zero-Shot CLIP (`02_clip_zeroshot.ipynb`)

Uses `openai/clip-vit-base-patch32` with **no gradient updates**. We ensemble 5 prompt templates per species (e.g. `"A close-up macro shot of a {species} butterfly"`) to create robust text anchors and compare via cosine similarity.

**Result: 28.2% Top-1.** CLIP knows what a butterfly looks like, but cannot distinguish 100 species whose differences live in wing venation details outside its training distribution.

### Approach 2 — Fine-Tuned EfficientNet-B0 (`01_efficientnet_finetuning.ipynb`)

Two-phase transfer learning on `efficientnet_b0` (timm, ImageNet pre-trained):

| Phase | Action | Trainable Params | Epochs | Val Acc |
|---|---|:---:|:---:|:---:|
| 1 — Linear Probe | Freeze backbone; train head only | 0.13 M | 5 | ~89.8% |
| 2 — Last Block Fine-Tune | Unfreeze `blocks.6` + head | 1.27 M (31%) | 10 | ~96.0% |

**Result: 96.6% Top-1, 99.4% Top-3.** Matching full fine-tune accuracy (all 4.14 M params) at 3× lower cost.

Grad-CAM confirms the model focuses on wing patterns and thorax — not the background.

---

## Bugs Fixed vs Original

| Location | Bug | Fix |
|---|---|---|
| `app.py` | `st.image(..., width="stretch")` — invalid argument, crashes at runtime | `use_container_width=True` |
| `app.py` | `st.dataframe(..., width="stretch")` — same issue | `use_container_width=True` |
| `data/species_metadata.json` | Key `"AMERICAN SNOUT"` didn't match class name `"AMERICAN SNOOT"` | Renamed key |
| `data/species_metadata.json` | Wiki summary facts were a single run-on string | Each numbered fact now on its own line |

---

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (model weights are ~16 MB, within the 100 MB file limit).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set: Repository = `Anurag-Kacholiya/Butterfly-Species-Detection`, Branch = `main`, Main file = `app.py`.
4. Click **Deploy** — live in ~3 minutes at a `*.streamlit.app` URL.

> If the `.pth` file exceeds GitHub's limit in future, track it with Git LFS:
> ```bash
> git lfs install
> git lfs track "model/*.pth"
> git add .gitattributes
> ```
