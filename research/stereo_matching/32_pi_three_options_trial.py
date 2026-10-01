"""
32_pi_three_options_trial.py - compare Pi-oriented stereo anti-spoof options.

This research runner tests three practical directions on the same left/right
pair:
  1. OpenCV SGBM as the first Pi runtime candidate.
  2. HITNet ONNX as a lighter deep candidate.
  3. Hybrid scoring: SGBM as runtime, stronger deep output as offline reference.

The script writes per-option outputs plus a summary JSON and a comparison image.
It does not touch the production attendance pipeline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "research" / "stereo_matching" / "31_deep_stereo_pipeline.py"


def run_case(name: str, out_base: Path, left: str, right: str, extra_args: list[str]) -> dict[str, Any]:
    out_dir = out_base / name
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--left",
        left,
        "--right",
        right,
        "--out",
        str(out_dir),
        *extra_args,
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "name": name,
            "ok": False,
            "elapsed_sec": elapsed,
            "output": proc.stdout[-3000:],
        }

    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report["name"] = name
    report["ok"] = True
    report["elapsed_sec"] = elapsed
    report["out_dir"] = str(out_dir)
    return report


def load_face_disparity(out_base: Path, case_name: str) -> np.ndarray:
    return np.load(out_base / case_name / "disparity_face.npy")


def relief_score(disp: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(disp)
    if finite.sum() < 100:
        return {
            "valid_ratio": float(finite.mean()),
            "relief_p95_p05": None,
            "center_edge_delta": None,
            "flatness_inverse": None,
        }

    vals = disp[finite]
    h, w = disp.shape
    cx0, cx1 = int(w * 0.40), int(w * 0.60)
    cy0, cy1 = int(h * 0.30), int(h * 0.62)
    center = disp[cy0:cy1, cx0:cx1]
    center_vals = center[np.isfinite(center)]
    edge_mask = finite.copy()
    edge_mask[:, cx0:cx1] = False
    edge_vals = disp[edge_mask]
    relief = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    center_edge = None
    if center_vals.size > 50 and edge_vals.size > 50:
        center_edge = float(np.nanmedian(center_vals) - np.nanmedian(edge_vals))

    return {
        "valid_ratio": float(finite.mean()),
        "relief_p95_p05": relief,
        "center_edge_delta": center_edge,
        "flatness_inverse": float(relief / max(abs(np.nanmedian(vals)), 1e-6)),
    }


def compare_runtime_to_reference(out_base: Path) -> dict[str, Any]:
    hybrid: dict[str, Any] = {
        "name": "04_hybrid_sgbm_runtime_raft_reference",
        "ok": True,
    }
    sgbm = load_face_disparity(out_base, "01_sgbm_pi_fast")
    ref = load_face_disparity(out_base, "03_deep_ref_raft_r4")
    hybrid["runtime_candidate"] = relief_score(sgbm)
    hybrid["offline_reference"] = relief_score(ref)

    ref_resized = cv2.resize(ref, (sgbm.shape[1], sgbm.shape[0]), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(sgbm) & np.isfinite(ref_resized)
    hybrid["overlap_valid_ratio"] = float(valid.mean())
    if valid.sum() > 100:
        sv = sgbm[valid]
        rv = ref_resized[valid]
        sv = (sv - np.percentile(sv, 5)) / max(np.percentile(sv, 95) - np.percentile(sv, 5), 1e-6)
        rv = (rv - np.percentile(rv, 5)) / max(np.percentile(rv, 95) - np.percentile(rv, 5), 1e-6)
        hybrid["rough_sgbm_vs_raft_corr"] = float(np.corrcoef(sv, rv)[0, 1])
    else:
        hybrid["rough_sgbm_vs_raft_corr"] = None
    return hybrid


def make_comparison_preview(out_base: Path) -> None:
    cases = [
        ("SGBM Pi fast", out_base / "01_sgbm_pi_fast" / "depth_color.png"),
        ("HITNet light", out_base / "02_deep_light_hitnet" / "depth_color.png"),
        ("RAFT ref", out_base / "03_deep_ref_raft_r4" / "depth_color.png"),
    ]
    panels = []
    target_h = 360
    for label, path in cases:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = target_h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        panels.append(img)
    if panels:
        cv2.imwrite(str(out_base / "comparison_preview.png"), cv2.hconcat(panels))


def compact_case(report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    return {
        "name": report["name"],
        "ok": True,
        "elapsed_sec": report["elapsed_sec"],
        "backend": report["backend"],
        "face_disparity": report.get("face_disparity"),
        "face_depth": report.get("face_depth"),
        "out_dir": report["out_dir"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare three Pi-oriented stereo liveness options")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", default="logs/stereo_trials/pi_three_options")
    ap.add_argument("--baseline-mm", type=float, default=100.0)
    args = ap.parse_args()

    out_base = (ROOT / args.out).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    common = [
        "--face-roi",
        "--face-mask",
        "--face-expand",
        "0.08",
        "--baseline-mm",
        str(args.baseline_mm),
    ]
    cases = [
        (
            "01_sgbm_pi_fast",
            [
                "--backend",
                "sgbm",
                *common,
                "--max-disp",
                "192",
                "--block-size",
                "9",
                "--work-scale",
                "0.5",
            ],
        ),
        (
            "02_deep_light_hitnet",
            [
                "--backend",
                "onnx",
                "--model",
                "models/deep_stereo/hitnet/eth3d/saved_model_240x320/model_float32.onnx",
                "--model-width",
                "320",
                "--model-height",
                "240",
                "--onnx-layout",
                "concat-gray-channel",
                "--input-scale",
                "unit",
                "--disparity-postprocess",
                "auto",
                *common,
            ],
        ),
        (
            "03_deep_ref_raft_r4",
            [
                "--backend",
                "onnx",
                "--model",
                "models/deep_stereo/raft_stereo_384x1280_r4.onnx",
                "--model-width",
                "1280",
                "--model-height",
                "384",
                "--onnx-layout",
                "two-input",
                "--input-scale",
                "raw255",
                "--disparity-postprocess",
                "abs",
                *common,
            ],
        ),
    ]

    reports = [run_case(name, out_base, args.left, args.right, extra) for name, extra in cases]
    summary = [compact_case(report) for report in reports]
    try:
        summary.append(compare_runtime_to_reference(out_base))
    except Exception as exc:
        summary.append({"name": "04_hybrid_sgbm_runtime_raft_reference", "ok": False, "error": str(exc)})

    make_comparison_preview(out_base)
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
