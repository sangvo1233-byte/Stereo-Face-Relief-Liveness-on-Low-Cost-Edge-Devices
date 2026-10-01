"""Tai model can thiet cho Face Attendance System.

- InsightFace goi `buffalo_l` (det_10g + w600k_r50 ArcFace 512-D + cac model phu).
  InsightFace tu tai ve MODELS_DIR khi FaceAnalysis khoi tao lan dau, nhung
  script nay tai truoc de chuan bi offline / kiem tra.
- MediaPipe `face_landmarker.task` (cho stream liveness / challenge).

Dung:
    python scripts/download_models.py
    python scripts/download_models.py --skip-insightface   # chi tai landmarker
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_landmarker(dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, "face_landmarker.task")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[skip] da co {out} ({os.path.getsize(out)} bytes)")
        return
    print(f"[tai ] face_landmarker.task <- {LANDMARKER_URL}")
    urllib.request.urlretrieve(LANDMARKER_URL, out)
    print(f"       xong {os.path.getsize(out)} bytes")


def download_insightface(model_root: str, name: str) -> None:
    """Goi InsightFace de no tu tai goi model ve model_root."""
    print(f"[tai ] InsightFace goi '{name}' -> {model_root}")
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        print(f"[loi ] chua cai insightface: {exc}")
        print("       chay: pip install -r requirements.txt")
        return
    app = FaceAnalysis(name=name, root=model_root,
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
    print("       xong (goi model da san sang)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download models for Face Attendance")
    ap.add_argument("--skip-insightface", action="store_true")
    ap.add_argument("--skip-landmarker", action="store_true")
    args = ap.parse_args()

    model_root = str(config.MODELS_DIR)
    if not args.skip_landmarker:
        download_landmarker(model_root)
    if not args.skip_insightface:
        download_insightface(model_root, config.INSIGHTFACE_MODEL)
    print("\nXong. Model nam trong:", model_root)


if __name__ == "__main__":
    main()
