"""
🦋 Butterfly Species Identifier — Streamlit App
================================================
Models: served from HuggingFace Hub (PyTorch / timm)
  - A: Linear Probe
  - B: Last Block Fine-Tune  [default]
  - C: Full Fine-Tune
Metadata: cached locally in data/species_metadata.json
"""

import json
import io
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🦋 Butterfly Identifier",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — set your HuggingFace repo here (or via env var HF_REPO)
# ══════════════════════════════════════════════════════════════════════════════
HF_REPO = os.environ.get("HF_REPO", "Anurag29104/butterfly-classifier")
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Model registry: display label → metadata
MODEL_REGISTRY = {
    "🔬 A — Linear Probe (92.20%)": {
        "type":       "pytorch",
        "hf_file":    "efficientnet_linear_probe.pth",
        "top1":       "92.20%",
        "top3":       "~95.50%",
        "macro_f1":   "~89.50%",
        "params":     "1.3K trained",
        "epochs":     "15",
        "note":       None,
    },
    "⭐ B — Last Block Fine-Tune (97.00%) [Default]": {
        "type":       "pytorch",
        "hf_file":    "efficientnet_lastblock_ft.pth",
        "top1":       "97.00%",
        "top3":       "99.20%",
        "macro_f1":   "96.92%",
        "params":     "2.9M trained",
        "epochs":     "15",
        "note":       None,
    },
    "🔥 C — Full Fine-Tune (97.00%)": {
        "type":       "pytorch",
        "hf_file":    "efficientnet_full_ft.pth",
        "top1":       "97.00%",
        "top3":       "99.20%",
        "macro_f1":   "96.87%",
        "params":     "5.3M trained",
        "epochs":     "15",
        "note":       None,
    },
}

DEFAULT_MODEL = "⭐ B — Last Block Fine-Tune (97.00%) [Default]"

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
CLASSES_PATH = DATA_DIR / "class_names.json"
CACHE_PATH   = DATA_DIR / "species_metadata.json"

# ── IUCN badge colours ──────────────────────────────────────────────────────────
IUCN_META = {
    "Least Concern":         {"color": "#27ae60", "icon": "🟢", "short": "LC"},
    "Near Threatened":       {"color": "#f39c12", "icon": "🟡", "short": "NT"},
    "Vulnerable":            {"color": "#e67e22", "icon": "🟠", "short": "VU"},
    "Endangered":            {"color": "#e74c3c", "icon": "🔴", "short": "EN"},
    "Critically Endangered": {"color": "#8e44ad", "icon": "🟣", "short": "CR"},
    "Data Deficient":        {"color": "#7f8c8d", "icon": "⚫", "short": "DD"},
    "Not Evaluated":         {"color": "#95a5a6", "icon": "⚪", "short": "NE"},
}

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%);
    color: #e8e8f0;
}

.hero-banner {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(74,144,217,0.15), rgba(80,200,120,0.10));
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4A90D9, #50C878, #F0B429);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub { font-size: 1.1rem; color: #8899aa; margin-top: 0.5rem; }

.upload-zone {
    border: 2px dashed rgba(74,144,217,0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(74,144,217,0.05);
}

.pred-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.pred-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(74,144,217,0.15); }
.pred-rank { font-size: 1.5rem; font-weight: 700; color: #4A90D9; }
.pred-name { font-size: 1.05rem; font-weight: 600; color: #e8e8f0; }
.pred-conf { font-size: 0.85rem; color: #7a8a9a; }

.iucn-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
}

.info-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #4A90D9;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    line-height: 1.7;
    font-size: 0.95rem;
    color: #ccd0d8;
}

.fun-fact {
    background: linear-gradient(135deg, rgba(80,200,120,0.12), rgba(74,144,217,0.08));
    border: 1px solid rgba(80,200,120,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.92rem;
    color: #b8e4c8;
}
.fun-fact::before { content: "💡  "; }

.conf-track { background: rgba(255,255,255,0.08); border-radius: 8px; height: 8px; margin-top: 6px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #4A90D9;
    border-bottom: 1px solid rgba(74,144,217,0.3);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}

.metric-pill {
    display: inline-block;
    background: rgba(74,144,217,0.15);
    border: 1px solid rgba(74,144,217,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.82rem;
    color: #7ab8e8;
    margin: 0.15rem;
}

.model-card {
    background: rgba(74,144,217,0.08);
    border: 1px solid rgba(74,144,217,0.25);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached by model key) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pytorch_model(hf_file: str, num_classes: int):
    """Download PyTorch weights from HuggingFace and load into EfficientNet-B0."""
    url = f"{HF_BASE}/{hf_file}"
    with st.spinner(f"⬇️ Downloading {hf_file} from HuggingFace…"):
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    state = torch.load(buf, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model



@st.cache_data(show_spinner=False)
def load_assets():
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    with open(CACHE_PATH) as f:
        species_meta = json.load(f)
    return class_names, species_meta


# ── Inference ──────────────────────────────────────────────────────────────────
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, image: Image.Image, class_names: list):
    """PyTorch inference — returns top-3 (class_name, probability) pairs."""
    tensor = EVAL_TF(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs  = logits.softmax(dim=-1).squeeze().numpy()
    top3   = probs.argsort()[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in top3]


# ── Helpers ────────────────────────────────────────────────────────────────────
def iucn_badge_html(status: str) -> str:
    meta  = IUCN_META.get(status, IUCN_META["Not Evaluated"])
    color = meta["color"]
    icon  = meta["icon"]
    return (f'<span class="iucn-badge" style="background:{color}22;'
            f'border:1px solid {color};color:{color};">{icon} {status}</span>')


def conf_bar_html(conf: float, color: str = "#4A90D9") -> str:
    pct = conf * 100
    return (f'<div class="conf-track"><div class="conf-fill" '
            f'style="width:{pct:.1f}%;background:{color};"></div></div>')


def get_species_img(species_name: str, meta: dict):
    entry   = meta.get(species_name, {})
    img_rel = entry.get("image_url", "")
    if img_rel:
        candidate = DATA_DIR / img_rel
        if candidate.exists():
            try:
                return Image.open(candidate).convert("RGB")
            except Exception:
                pass
    return None


RANK_COLORS = ["#F0B429", "#C0C0C0", "#CD7F32"]
RANK_EMOJI  = ["🥇", "🥈", "🥉"]

# ── Load static assets ─────────────────────────────────────────────────────────
try:
    class_names, species_meta = load_assets()
    assets_ok = True
except Exception as e:
    st.error(f"Failed to load metadata/class names: {e}")
    assets_ok = False
    class_names, species_meta = [], {}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🦋 Butterfly Identifier")
    st.markdown("---")

    # ── Model selector ───────────────────────────────────────────────────────
    st.markdown("**🔬 Select Model**")
    model_choice = st.radio(
        "Choose an architecture:",
        list(MODEL_REGISTRY.keys()),
        index=list(MODEL_REGISTRY.keys()).index(DEFAULT_MODEL),
        label_visibility="collapsed",
    )
    cfg = MODEL_REGISTRY[model_choice]

    # Dynamic metrics pills
    st.markdown("**Performance**")
    st.markdown(
        f'<span class="metric-pill">Top-1: {cfg["top1"]}</span>'
        f'<span class="metric-pill">Top-3: {cfg["top3"]}</span>'
        f'<span class="metric-pill">F1: {cfg["macro_f1"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("**Architecture**")
    st.markdown(
        f'<span class="metric-pill">{cfg["params"]}</span>'
        f'<span class="metric-pill">Epochs: {cfg["epochs"]}</span>',
        unsafe_allow_html=True,
    )
    if cfg["note"]:
        st.info(cfg["note"])

    st.markdown("---")
    st.markdown("**About**")
    st.markdown(
        "Upload a butterfly photo to get:\n"
        "- Top-3 species predictions\n"
        "- IUCN conservation status\n"
        "- Wikipedia summary\n"
        "- Fun facts"
    )
    st.markdown("---")
    st.markdown("**IUCN Legend**")
    for status, meta in IUCN_META.items():
        if status == "Not Evaluated":
            continue
        st.markdown(f'{meta["icon"]} **{meta["short"]}** — {status}')
    st.markdown("---")
    st.caption(f"HF Repo: `{HF_REPO}`")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🦋 Butterfly Species Identifier</div>
    <div class="hero-sub">
        Upload a field photo → Top-3 predictions · IUCN status · Fun facts
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
upload_col, results_col = st.columns([1, 1.6], gap="large")

with upload_col:
    st.markdown(
        f'<div class="section-header">📷 Upload Your Photo'
        f'<br><small style="color:#7ab8e8;font-weight:400;font-size:0.8rem;">'
        f'Model: {model_choice.split("—")[0].strip()}</small></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        label="Drop a butterfly image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        user_img = Image.open(uploaded).convert("RGB")
        st.image(user_img, caption="Uploaded photo", width='stretch')
    else:
        st.markdown("""
        <div class="upload-zone">
            <p style="font-size:2.5rem;margin:0;">📷</p>
            <p style="color:#5a6a7a;margin:0.5rem 0 0;">
                Drag &amp; drop a butterfly photo<br>
                <small>JPG · PNG · WEBP</small>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Inference ──────────────────────────────────────────────────────────────────
with results_col:
    if not uploaded or not assets_ok:
        st.markdown('<div class="section-header">🔍 Results</div>', unsafe_allow_html=True)
        st.info("Upload a butterfly photo on the left to see predictions.")

    else:
        # Load model on demand (cached after first download)
        try:
            active_model = load_pytorch_model(cfg["hf_file"], len(class_names))
        except Exception as e:
            st.error(
                f"Could not load **{cfg['hf_file']}** from HuggingFace.\n\n"
                f"Make sure you've run `upload_to_hf.py` and set `HF_REPO` correctly.\n\n"
                f"Error: `{e}`"
            )
            st.stop()

        with st.spinner("Analysing image…"):
            top3 = predict(active_model, user_img, class_names)

        st.markdown('<div class="section-header">🔍 Top-3 Predictions</div>', unsafe_allow_html=True)
        for rank, (species, conf) in enumerate(top3):
            color = RANK_COLORS[rank]
            emoji = RANK_EMOJI[rank]
            bar   = conf_bar_html(conf, color)
            st.markdown(f"""
            <div class="pred-card">
                <span class="pred-rank">{emoji}</span>
                <span class="pred-name" style="margin-left:0.6rem;">{species.title()}</span>
                <span class="pred-conf" style="float:right;margin-top:4px;">{conf*100:.1f}%</span>
                {bar}
            </div>
            """, unsafe_allow_html=True)

        top_species = top3[0][0]
        top_conf    = top3[0][1]
        meta_entry  = species_meta.get(top_species, {})

        if top_conf < 0.5:
            st.warning(
                f"⚠️ Low confidence ({top_conf*100:.1f}%). "
                "This might be an unusual angle, damaged wing, or a species outside the 100 trained classes."
            )

        st.divider()
        info_left, info_right = st.columns([1.1, 1], gap="medium")

        with info_left:
            iucn = meta_entry.get("iucn_status", "Not Evaluated")
            st.markdown('<div class="section-header" style="font-size:1rem;">🛡️ IUCN Status</div>',
                        unsafe_allow_html=True)
            st.markdown(iucn_badge_html(iucn), unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">📖 Wikipedia</div>',
                        unsafe_allow_html=True)
            wiki = meta_entry.get("wiki_summary", "No summary available.")
            st.markdown(f'<div class="info-box">{wiki.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            fun_facts = meta_entry.get("fun_facts", [])
            if fun_facts:
                st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">✨ Fun Facts</div>',
                            unsafe_allow_html=True)
                for fact in fun_facts:
                    st.markdown(f'<div class="fun-fact">{fact}</div>', unsafe_allow_html=True)

        with info_right:
            ref_img = get_species_img(top_species, species_meta)
            if ref_img:
                st.markdown('<div class="section-header" style="font-size:1rem;">🖼️ Reference Photo</div>',
                            unsafe_allow_html=True)
                st.image(ref_img, caption=f"{top_species.title()} (reference)", width='stretch')
            else:
                st.markdown("*No reference image available.*")

            st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">📊 Confidence</div>',
                        unsafe_allow_html=True)
            for rank, (sp, cf) in enumerate(top3):
                iucn_sp   = species_meta.get(sp, {}).get("iucn_status", "N/A")
                iucn_info = IUCN_META.get(iucn_sp, IUCN_META["Not Evaluated"])
                st.markdown(
                    f"**{RANK_EMOJI[rank]} {sp.title()}**  \n"
                    f"{iucn_info['icon']} {iucn_sp} &nbsp;|&nbsp; `{cf*100:.2f}%`"
                )
                st.progress(float(cf), text="")


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📈 Full Ablation Study Results", expanded=False):
    m1, m2, m3, m4, m5 = st.columns(5)
    # Show metrics for currently selected model
    m1.metric("Top-1 Accuracy",  cfg["top1"])
    m2.metric("Top-3 Accuracy",  cfg["top3"])
    m3.metric("Macro F1",        cfg["macro_f1"])
    m4.metric("Trainable Params", cfg["params"])
    m5.metric("Epochs",           cfg["epochs"])

    st.markdown("#### All Models Compared")
    ablation_data = {
        "Method": [
            "Baseline (Author H5, Keras)",
            "A — Linear Probe",
            "B — Fine-Tune Last Block",
            "C — Full Fine-Tune",
            "D — CLIP Zero-Shot (Ensemble)",
        ],
        "Trainable Params": ["0", "1.3k", "2.9M", "5.3M", "0 / Frozen"],
        "Epochs":           ["0", "15", "15", "15", "0"],
        "Test Top-1":       ["97.60%", "92.20%", "97.00%", "97.00%", "30.80%"],
        "Test Top-3":       ["99.60%", "~95.50%", "99.20%", "99.20%", "44.20%"],
        "Macro F1":         ["~97.50%", "~89.50%", "96.92%", "96.87%", "24.57%"],
        "HF Weights":       ["n/a (Keras)", "efficientnet_linear_probe.pth",
                             "efficientnet_lastblock_ft.pth", "efficientnet_full_ft.pth",
                             "openai/clip-vit-base-patch32"],
    }
    st.dataframe(ablation_data, hide_index=True, width='stretch')
    st.caption(f"Weights hosted at: https://huggingface.co/{HF_REPO}")
