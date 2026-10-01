"""
34_synthetic_spoof_benchmark.py - benchmark stereo liveness without a camera rig.

This script creates synthetic stereo pairs from one left image:
  - flat fronto-parallel print/screen
  - tilted planar print/screen
  - curved print/screen

It can also include an existing real stereo pair as the positive reference. The
goal is not to replace a physical camera benchmark; it is a pre-benchmark for
checking whether the depth scoring logic can separate face-like relief from
planar or curved spoof geometry before the stereo rig is available.
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


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    return img


def make_right_from_disparity(left: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Create a synthetic right frame from a left frame and left-view disparity."""
    h, w = left.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # For an output right pixel x_r, sample approximately from x_l = x_r + d.
    map_x = xx + disp.astype(np.float32)
    map_y = yy
    return cv2.remap(left, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_disparity(shape: tuple[int, int], kind: str, base: float, amp: float) -> np.ndarray:
    h, w = shape
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, w), np.linspace(-1.0, 1.0, h))
    if kind == "flat_front":
        disp = np.full((h, w), base, dtype=np.float32)
    elif kind == "flat_tilt_x":
        disp = base + amp * xx
    elif kind == "flat_tilt_xy":
        disp = base + amp * (0.75 * xx + 0.25 * yy)
    elif kind == "curved_cylinder":
        disp = base + amp * (1.0 - np.clip(xx**2, 0, 1))
    elif kind == "curved_soft":
        disp = base + amp * np.exp(-((xx / 0.75) ** 2 + (yy / 1.10) ** 2))
    else:
        raise SystemExit(f"Unknown synthetic kind: {kind}")
    return np.maximum(disp.astype(np.float32), 1.0)


def run_pipeline(
    left: str,
    right: str,
    out_dir: Path,
    backend: str,
    model: str | None,
    model_width: int | None,
    model_height: int | None,
    onnx_layout: str,
    input_scale: str,
    disparity_postprocess: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PIPELINE),
        "--left",
        left,
        "--right",
        right,
        "--out",
        str(out_dir),
        "--backend",
        backend,
        "--face-roi",
        "--face-mask",
        "--face-expand",
        "0.08",
        "--baseline-mm",
        "100",
    ]
    if backend == "onnx":
        if not model:
            raise SystemExit("--backend onnx requires --model")
        cmd += [
            "--model",
            model,
            "--onnx-layout",
            onnx_layout,
            "--input-scale",
            input_scale,
            "--disparity-postprocess",
            disparity_postprocess,
        ]
        if model_width:
            cmd += ["--model-width", str(model_width)]
        if model_height:
            cmd += ["--model-height", str(model_height)]
    else:
        cmd += ["--max-disp", "192", "--block-size", "9", "--work-scale", "0.5"]

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return {"ok": False, "elapsed_sec": elapsed, "output": proc.stdout[-3000:]}
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report["ok"] = True
    report["elapsed_sec"] = elapsed
    return report


def fit_plane_residual(arr: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(arr)
    if finite.sum() < 100:
        return {"valid_ratio": float(finite.mean()), "plane_rmse": None, "relief_p95_p05": None}
    y, x = np.where(finite)
    z = arr[finite].astype(np.float64)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = (x - x.mean()) / max(x.std(), 1.0)
    y = (y - y.mean()) / max(y.std(), 1.0)
    a = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    pred = a @ coef
    residual = z - pred
    relief = float(np.percentile(z, 95) - np.percentile(z, 5))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "valid_ratio": float(finite.mean()),
        "plane_rmse": rmse,
        "relief_p95_p05": relief,
        "plane_rmse_over_relief": float(rmse / max(relief, 1e-6)),
    }


def score_output(out_dir: Path) -> dict[str, Any]:
    disp_path = out_dir / "disparity_face.npy"
    if not disp_path.exists():
        return {"ok": False, "error": "missing disparity_face.npy"}
    disp = np.load(disp_path)
    stats = fit_plane_residual(disp)
    plane_rmse = stats.get("plane_rmse") or 0.0
    relief = stats.get("relief_p95_p05") or 0.0
    # Synthetic flat cases can have a large ratio because relief is near zero.
    # Use absolute residual relief after plane fitting as the first signal.
    stats["decision_hint"] = "more_3d_like" if plane_rmse >= 6.0 and relief >= 20.0 else "more_planar_like"
    return stats


def make_preview_grid(out_base: Path, rows: list[dict[str, Any]]) -> None:
    panels = []
    target_h = 260
    for row in rows:
        img_path = Path(row["out_dir"]) / "depth_color.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        scale = target_h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, row["name"], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        panels.append(img)
    if panels:
        cv2.imwrite(str(out_base / "synthetic_spoof_comparison.png"), cv2.hconcat(panels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic spoof benchmark without a physical stereo rig")
    ap.add_argument("--left", required=True)
    ap.add_argument("--real-right", default=None, help="optional real right image for positive reference")
    ap.add_argument("--out", default="logs/stereo_trials/synthetic_spoof_benchmark")
    ap.add_argument("--backend", choices=["onnx", "sgbm"], default="onnx")
    ap.add_argument("--model", default="models/deep_stereo/raft_stereo_384x1280_r4.onnx")
    ap.add_argument("--model-width", type=int, default=1280)
    ap.add_argument("--model-height", type=int, default=384)
    ap.add_argument("--onnx-layout", default="two-input")
    ap.add_argument("--input-scale", default="raw255")
    ap.add_argument("--disparity-postprocess", default="abs")
    ap.add_argument("--base-disp", type=float, default=180.0)
    ap.add_argument("--amp-disp", type=float, default=45.0)
    args = ap.parse_args()

    out_base = (ROOT / args.out).resolve()
    pairs_dir = out_base / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    left = read_image(args.left)
    left_path = pairs_dir / "left.png"
    cv2.imwrite(str(left_path), left)

    rows: list[dict[str, Any]] = []
    cases = ["flat_front", "flat_tilt_x", "flat_tilt_xy", "curved_cylinder", "curved_soft"]
    for case in cases:
        disp = make_disparity(left.shape[:2], case, args.base_disp, args.amp_disp)
        right = make_right_from_disparity(left, disp)
        right_path = pairs_dir / f"{case}_right.png"
        np.save(pairs_dir / f"{case}_true_disparity.npy", disp)
        cv2.imwrite(str(right_path), right)
        case_out = out_base / case
        report = run_pipeline(
            str(left_path),
            str(right_path),
            case_out,
            args.backend,
            args.model,
            args.model_width,
            args.model_height,
            args.onnx_layout,
            args.input_scale,
            args.disparity_postprocess,
        )
        row = {
            "name": case,
            "label": "synthetic_spoof",
            "out_dir": str(case_out),
            "pipeline": report,
            "score": score_output(case_out) if report.get("ok") else None,
        }
        rows.append(row)

    if args.real_right:
        real_out = out_base / "real_pair_reference"
        report = run_pipeline(
            args.left,
            args.real_right,
            real_out,
            args.backend,
            args.model,
            args.model_width,
            args.model_height,
            args.onnx_layout,
            args.input_scale,
            args.disparity_postprocess,
        )
        rows.insert(
            0,
            {
                "name": "real_pair_reference",
                "label": "real_or_capture_reference",
                "out_dir": str(real_out),
                "pipeline": report,
                "score": score_output(real_out) if report.get("ok") else None,
            },
        )

    make_preview_grid(out_base, rows)
    (out_base / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = [
        {
            "name": row["name"],
            "label": row["label"],
            "elapsed_sec": row["pipeline"].get("elapsed_sec"),
            "face_disparity": row["pipeline"].get("face_disparity"),
            "score": row["score"],
        }
        for row in rows
    ]
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
