"""
🦋 Butterfly Species Identifier — Streamlit App
================================================
Model   : EfficientNet-B0 (fine-tuned, timm)  — model/efficientnet_butterfly.pth
Classes : data/class_names.json (100 species)
Metadata: data/species_metadata.json (wiki + IUCN + fun facts)
Images  : data/species_images/*.jpg
"""

import json
import os
from pathlib import Path

import numpy as np
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

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
MODEL_PATH   = BASE_DIR / "model" / "efficientnet_butterfly.pth"
CLASSES_PATH = DATA_DIR / "class_names.json"
CACHE_PATH   = DATA_DIR / "species_metadata.json"

# ── IUCN badge colours ──────────────────────────────────────────────────────────
IUCN_META = {
    "Least Concern":  {"color": "#27ae60", "icon": "🟢", "short": "LC"},
    "Near Threatened":{"color": "#f39c12", "icon": "🟡", "short": "NT"},
    "Vulnerable":     {"color": "#e67e22", "icon": "🟠", "short": "VU"},
    "Endangered":     {"color": "#e74c3c", "icon": "🔴", "short": "EN"},
    "Critically Endangered": {"color": "#8e44ad", "icon": "🟣", "short": "CR"},
    "Data Deficient": {"color": "#7f8c8d", "icon": "⚫", "short": "DD"},
    "Not Evaluated":  {"color": "#95a5a6", "icon": "⚪", "short": "NE"},
}

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f0c29 100%);
    color: #e8e8f0;
}

/* Hero banner */
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
.hero-sub {
    font-size: 1.1rem;
    color: #8899aa;
    margin-top: 0.5rem;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed rgba(74,144,217,0.5);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(74,144,217,0.05);
    transition: border-color 0.3s;
}

/* Prediction cards */
.pred-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.pred-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(74,144,217,0.15);
}
.pred-rank {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4A90D9;
}
.pred-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e8e8f0;
}
.pred-conf {
    font-size: 0.85rem;
    color: #7a8a9a;
}

/* IUCN badge */
.iucn-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
}

/* Info section */
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

/* Fun fact */
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

/* Confidence bar track */
.conf-track {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #4A90D9;
    border-bottom: 1px solid rgba(74,144,217,0.3);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Metric pills */
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
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B0 model…")
def load_model(model_path: Path, num_classes: int):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
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


# ── Inference transform ─────────────────────────────────────────────────────────
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, image: Image.Image, class_names: list):
    """Return sorted list of (class_name, probability) — top-3."""
    tensor = EVAL_TF(image.convert("RGB")).unsqueeze(0)
    logits = model(tensor)
    probs  = logits.softmax(dim=-1).squeeze().numpy()
    top3   = probs.argsort()[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in top3]


# ── Helpers ─────────────────────────────────────────────────────────────────────
def iucn_badge_html(status: str) -> str:
    meta  = IUCN_META.get(status, IUCN_META["Not Evaluated"])
    color = meta["color"]
    icon  = meta["icon"]
    return (
        f'<span class="iucn-badge" style="background:{color}22;'
        f'border:1px solid {color};color:{color};">'
        f'{icon} {status}</span>'
    )


def conf_bar_html(conf: float, color: str = "#4A90D9") -> str:
    pct = conf * 100
    return (
        f'<div class="conf-track">'
        f'<div class="conf-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
    )


RANK_COLORS = ["#F0B429", "#C0C0C0", "#CD7F32"]   # gold / silver / bronze
RANK_EMOJI  = ["🥇", "🥈", "🥉"]


def get_species_img(species_name: str, meta: dict):
    """Return PIL image for the species reference photo, or None."""
    entry = meta.get(species_name, {})
    img_rel = entry.get("image_url", "")
    if img_rel:
        candidate = DATA_DIR / img_rel   # e.g. data/species_images/adonis.jpg
        if candidate.exists():
            try:
                return Image.open(candidate).convert("RGB")
            except Exception:
                pass
    return None


# ── Load assets ─────────────────────────────────────────────────────────────────
try:
    class_names, species_meta = load_assets()
    model = load_model(MODEL_PATH, len(class_names))
    model_ok = True
except Exception as e:
    st.error(f"Failed to load model/assets: {e}")
    model_ok = False
    class_names, species_meta = [], {}


# ══════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🦋 Butterfly Identifier")
    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown(
        "Upload a butterfly photo to get:\n"
        "- Top-3 species predictions\n"
        "- IUCN conservation status\n"
        "- Wikipedia species summary\n"
        "- Fun facts"
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown(
        '<span class="metric-pill">EfficientNet-B0</span>'
        '<span class="metric-pill">timm</span>'
        '<span class="metric-pill">PyTorch</span>',
        unsafe_allow_html=True,
    )
    st.markdown("**Performance**")
    st.markdown(
        '<span class="metric-pill">Top-1: 96.6%</span>'
        '<span class="metric-pill">Top-3: 99.4%</span>'
        '<span class="metric-pill">100 species</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**IUCN Status Legend**")
    for status, meta in IUCN_META.items():
        if status in {"Not Evaluated"}:
            continue
        st.markdown(
            f'{meta["icon"]} **{meta["short"]}** — {status}',
        )
    st.markdown("---")
    st.caption("SMAI Mini-Project · EfficientNet-B0 Fine-Tuning")


# ══════════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🦋 Butterfly Species Identifier</div>
    <div class="hero-sub">
        Upload a field photo → Get the species name, conservation status &amp; a fun fact
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════════
upload_col, results_col = st.columns([1, 1.6], gap="large")

with upload_col:
    st.markdown('<div class="section-header">📷 Upload Your Photo</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="Drop a butterfly image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        user_img = Image.open(uploaded).convert("RGB")
        st.image(user_img, caption="Uploaded photo", use_container_width=True)
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


# ── Run inference when image uploaded ───────────────────────────────────────────
with results_col:
    if not uploaded or not model_ok:
        st.markdown('<div class="section-header">🔍 Results</div>',
                    unsafe_allow_html=True)
        st.info("Upload a butterfly photo on the left to see predictions.")

    else:
        with st.spinner("Analysing image…"):
            top3 = predict(model, user_img, class_names)

        # ── TOP-3 PREDICTIONS ────────────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Top-3 Predictions</div>',
                    unsafe_allow_html=True)

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

        # ══════════════════════════════════════════════════════════════════════
        # DETAILED INFO FOR TOP-1 PREDICTION
        # ══════════════════════════════════════════════════════════════════════
        top_species = top3[0][0]
        top_conf    = top3[0][1]
        meta_entry  = species_meta.get(top_species, {})

        # Confidence warning
        if top_conf < 0.5:
            st.warning(
                f"⚠️ Low confidence ({top_conf*100:.1f}%). "
                "This might be an unusual angle, damaged wing, or a species outside the 100 trained classes."
            )

        st.divider()

        # ── SPECIES DETAIL COLUMNS ───────────────────────────────────────────
        info_left, info_right = st.columns([1.1, 1], gap="medium")

        with info_left:
            # IUCN Badge
            iucn = meta_entry.get("iucn_status", "Not Evaluated")
            st.markdown('<div class="section-header" style="font-size:1rem;">🛡️ IUCN Conservation Status</div>',
                        unsafe_allow_html=True)
            st.markdown(iucn_badge_html(iucn), unsafe_allow_html=True)

            # Wikipedia summary
            st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">📖 Wikipedia Summary</div>',
                        unsafe_allow_html=True)
            wiki = meta_entry.get("wiki_summary", "No summary available.")
            wiki_html = wiki.replace("\n", "<br>")
            st.markdown(f'<div class="info-box">{wiki_html}</div>', unsafe_allow_html=True)

            # Fun facts
            fun_facts = meta_entry.get("fun_facts", [])
            if fun_facts:
                st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">✨ Fun Facts</div>',
                            unsafe_allow_html=True)
                for fact in fun_facts:
                    st.markdown(f'<div class="fun-fact">{fact}</div>',
                                unsafe_allow_html=True)

        with info_right:
            # Reference species image from cache
            ref_img = get_species_img(top_species, species_meta)
            if ref_img:
                st.markdown('<div class="section-header" style="font-size:1rem;">🖼️ Reference Photo</div>',
                            unsafe_allow_html=True)
                st.image(ref_img, caption=f"{top_species.title()} (reference)",
                         use_container_width=True)
            else:
                st.markdown("*No reference image available.*")

            # All-3 confidence breakdown as a mini table
            st.markdown('<div class="section-header" style="font-size:1rem;margin-top:1.2rem;">📊 Confidence Breakdown</div>',
                        unsafe_allow_html=True)
            for rank, (sp, cf) in enumerate(top3):
                iucn_sp   = species_meta.get(sp, {}).get("iucn_status", "N/A")
                iucn_info = IUCN_META.get(iucn_sp, IUCN_META["Not Evaluated"])
                st.markdown(
                    f"**{RANK_EMOJI[rank]} {sp.title()}**  \n"
                    f"{iucn_info['icon']} {iucn_sp} &nbsp;|&nbsp; `{cf*100:.2f}%`"
                )
                st.progress(float(cf), text="")


# ══════════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE STRIP (always visible at bottom)
# ══════════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("📈 Model Performance & Ablation Study", expanded=False):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Top-1 Accuracy",  "96.60%")
    m2.metric("Top-3 Accuracy",  "99.40%")
    m3.metric("Macro F1",        "96.45%")
    m4.metric("Macro Precision", "97.03%")
    m5.metric("Macro Recall",    "96.60%")

    st.markdown("#### Ablation Study Results")
    ablation_data = {
        "Method": [
            "Baseline (Author H5, Keras)",
            "A — Linear Probe",
            "B — Fine-Tune Last Block ✅",
            "C — Full Fine-Tune",
            "D — CLIP Zero-Shot (Ensemble)"
        ],
        "Trainable Params": ["0", "0.13M", "1.27M", "4.14M", "0 / Frozen"],
        "Epochs": ["0", "5", "5+10", "10", "0"],
        "Test Top-1": ["97.60%", "~89.80%", "96.60%", "96.60%", "30.80%"],
        "Test Top-3": ["99.60%", "~95.50%", "99.40%", "99.20%", "44.20%"],
        "Macro F1":   ["~97.50%", "~89.50%", "96.45%", "96.53%", "24.57%"],
    }
    st.dataframe(ablation_data, hide_index=True, use_container_width=True)
    st.caption("✅ = deployed model in this app")
