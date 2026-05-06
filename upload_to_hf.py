#!/usr/bin/env python3
"""
upload_to_hf.py
---------------
Run this AFTER downloading the model files from Kaggle.

Steps:
  1. Download all .pth files + the author .h5 from Kaggle to  model/
  2. Get a HuggingFace write token: https://huggingface.co/settings/tokens
  3. export HF_TOKEN=hf_xxxxxxxxxxxx
  4. pip install huggingface_hub
  5. python upload_to_hf.py

Files uploaded to HuggingFace:
  - efficientnet_linear_probe.pth
  - efficientnet_lastblock_ft.pth
  - efficientnet_full_ft.pth
  - EfficientNetB0-100-(224 X 224)- 97.59.h5   (author baseline)
  - class_names.json
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── CONFIG ─────────────────────────────────────────────────────────────────────
HF_REPO  = os.environ.get("HF_REPO",  "Anurag29104/butterfly-classifier")
HF_TOKEN = os.environ.get("HF_TOKEN", None)   # required — see steps above

# ── Files to upload (relative to this script) ─────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
DATA_DIR  = Path(__file__).parent / "data"

UPLOAD_MAP = {
    # local path → HF filename
    MODEL_DIR / "efficientnet_linear_probe.pth":            "efficientnet_linear_probe.pth",
    MODEL_DIR / "efficientnet_lastblock_ft.pth":            "efficientnet_lastblock_ft.pth",
    MODEL_DIR / "efficientnet_full_ft.pth":                 "efficientnet_full_ft.pth",
    MODEL_DIR / "EfficientNetB0-100-(224 X 224)- 97.59.h5": "EfficientNetB0-100-(224 X 224)- 97.59.h5",
    DATA_DIR  / "class_names.json":                         "class_names.json",

}


def main():
    # ── Auth check ───────────────────────────────────────────────────────────
    if not HF_TOKEN:
        print("❌  HF_TOKEN is not set.  Run one of:")
        print("    export HF_TOKEN=hf_xxxx        # get at https://huggingface.co/settings/tokens")
        print("    huggingface-cli login           # interactive login")
        print()
        raise SystemExit(1)

    api = HfApi(token=HF_TOKEN)

    # ── Create repo if needed ────────────────────────────────────────────────
    try:
        create_repo(HF_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
        print(f"✅ Repo ready: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"⚠️  create_repo warning (repo may already exist): {e}")

    # ── Upload files ─────────────────────────────────────────────────────────
    for local, remote in UPLOAD_MAP.items():
        if not local.exists():
            print(f"  ⚠️  SKIP (file not found locally): {local}")
            continue
        print(f"  ⬆️  Uploading {local.name} → {HF_REPO}/{remote} …")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=HF_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"     ✅ Done")

    print(f"\n🎉 All uploads complete!")
    print(f"   View at: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
