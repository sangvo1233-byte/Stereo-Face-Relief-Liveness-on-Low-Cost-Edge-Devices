"""
31_deep_stereo_pipeline.py - experimental stereo depth/disparity runner.

Purpose:
  Build a temporary pipeline for trying deep stereo models on left/right images.
  The production app does not use this file.

Backends:
  - onnx: generic ONNX deep stereo runner for two-input or concatenated-input models.
  - sgbm: OpenCV StereoSGBM fallback to test IO/output before a deep model exists.

Examples:
  py research/stereo_matching/31_deep_stereo_pipeline.py ^
      --left C:/Users/ADMIN/Desktop/1.jpg --right C:/Users/ADMIN/Desktop/2.jpg ^
      --backend sgbm --out logs/stereo_trials/sgbm_pair

  py research/stereo_matching/31_deep_stereo_pipeline.py ^
      --left left_rectified.png --right right_rectified.png ^
      --backend onnx --model models/deep_stereo/model.onnx ^
      --model-width 640 --model-height 384 --out logs/stereo_trials/deep_pair

Notes:
  Deep stereo only becomes meaningful after the camera pair is fixed, calibrated,
  rectified, and synchronized. If the input pair is not rectified, the generated
  map may look plausible but should not be trusted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    return img


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(float(p.strip())) for p in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("--roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("--roi must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def clip_roi(roi: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    x0 = max(0, min(w - 1, x0))
    x1 = max(1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(1, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("ROI is outside image")
    return x0, y0, x1, y1


def union_rois(
    rois: list[tuple[int, int, int, int]],
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    if not rois:
        raise SystemExit("No ROI to union")
    x0 = min(r[0] for r in rois)
    y0 = min(r[1] for r in rois)
    x1 = max(r[2] for r in rois)
    y1 = max(r[3] for r in rois)
    return clip_roi((x0, y0, x1, y1), w, h)


def detect_face_landmarks_mediapipe(img: np.ndarray, model_path: str) -> np.ndarray:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as exc:
        raise SystemExit(f"MediaPipe not available for face detection: {exc}") from exc

    h, w = img.shape[:2]
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mpimg)
    if not res.face_landmarks:
        raise SystemExit("No face detected")
    return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], dtype=np.float32)


def face_roi_from_landmarks(
    pts: np.ndarray,
    image_shape: tuple[int, int],
    expand: float,
) -> tuple[int, int, int, int]:
    h, w = image_shape
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    pad = expand * max(bw, bh)
    return clip_roi((int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad)), w, h)


def detect_face_roi_mediapipe(img: np.ndarray, model_path: str, expand: float) -> tuple[int, int, int, int]:
    pts = detect_face_landmarks_mediapipe(img, model_path)
    return face_roi_from_landmarks(pts, img.shape[:2], expand)


def face_mask_from_landmarks(
    pts: np.ndarray,
    shape: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
    dilate_px: int,
) -> np.ndarray:
    """Create a filled convex-hull face mask in crop coordinates."""
    h, w = shape
    offset_x = roi[0] if roi else 0
    offset_y = roi[1] if roi else 0
    shifted = pts.copy()
    shifted[:, 0] -= offset_x
    shifted[:, 1] -= offset_y
    shifted[:, 0] = np.clip(shifted[:, 0], 0, w - 1)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, h - 1)
    hull = cv2.convexHull(shifted.astype(np.int32))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if dilate_px > 0:
        k = max(1, int(dilate_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def transform_points(points: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, hmat)
    return warped.reshape(-1, 2)


def mean_abs_y_delta(left_pts: np.ndarray, right_pts: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        mask = mask.reshape(-1).astype(bool)
        left_pts = left_pts[mask]
        right_pts = right_pts[mask]
    if left_pts.size == 0 or right_pts.size == 0:
        return float("nan")
    return float(np.mean(np.abs(left_pts[:, 1] - right_pts[:, 1])))


def rectify_pair_from_face_landmarks(
    left: np.ndarray,
    right: np.ndarray,
    model_path: str,
    ransac_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    left_pts = detect_face_landmarks_mediapipe(left, model_path)
    right_pts = detect_face_landmarks_mediapipe(right, model_path)
    before_y = mean_abs_y_delta(left_pts, right_pts)
    fund, inlier_mask = cv2.findFundamentalMat(
        left_pts,
        right_pts,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_px,
        confidence=0.995,
    )
    if fund is None or fund.shape != (3, 3):
        raise SystemExit("Cannot estimate fundamental matrix from face landmarks")
    h, w = left.shape[:2]
    ok, h1, h2 = cv2.stereoRectifyUncalibrated(left_pts, right_pts, fund, imgSize=(w, h))
    if not ok:
        raise SystemExit("stereoRectifyUncalibrated failed")

    left_rect = cv2.warpPerspective(left, h1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_rect = cv2.warpPerspective(right, h2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    left_warped = transform_points(left_pts, h1)
    right_warped = transform_points(right_pts, h2)
    after_y = mean_abs_y_delta(left_warped, right_warped, inlier_mask)
    inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    meta = {
        "method": "face-landmark-fundamental-uncalibrated",
        "ransac_px": ransac_px,
        "landmarks": int(len(left_pts)),
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(left_pts), 1)),
        "mean_abs_y_delta_before_px": before_y,
        "mean_abs_y_delta_after_inliers_px": after_y,
    }
    return left_rect, right_rect, meta


def apply_mask_to_map(data: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return data
    out = data.copy()
    out[mask == 0] = np.nan
    return out


def apply_mask_to_bgr(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return img
    out = img.copy()
    out[mask == 0] = 0
    return out


def crop_pair(
    left: np.ndarray,
    right: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    if roi is None:
        return left, right, None
    h, w = left.shape[:2]
    roi = clip_roi(roi, w, h)
    x0, y0, x1, y1 = roi
    return left[y0:y1, x0:x1].copy(), right[y0:y1, x0:x1].copy(), roi


def resize_work_inputs(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if scale <= 0:
        raise SystemExit("--work-scale must be > 0")
    if abs(scale - 1.0) < 1e-6:
        return left, right, left_mask, right_mask
    h, w = left.shape[:2]
    out_w = max(16, int(round(w * scale)))
    out_h = max(16, int(round(h * scale)))
    size = (out_w, out_h)
    left_small = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    left_mask_small = None
    right_mask_small = None
    if left_mask is not None:
        left_mask_small = cv2.resize(left_mask, size, interpolation=cv2.INTER_NEAREST)
    if right_mask is not None:
        right_mask_small = cv2.resize(right_mask, size, interpolation=cv2.INTER_NEAREST)
    return left_small, right_small, left_mask_small, right_mask_small


def normalize_input(img: np.ndarray, size: tuple[int, int], input_scale: str) -> np.ndarray:
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    if input_scale == "unit":
        rgb /= 255.0
    elif input_scale == "raw255":
        pass
    else:
        raise SystemExit(f"Unknown input scale: {input_scale}")
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return chw.astype(np.float32)


def normalize_gray_input(img: np.ndarray, size: tuple[int, int], input_scale: str) -> np.ndarray:
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if input_scale == "unit":
        gray /= 255.0
    elif input_scale == "raw255":
        pass
    else:
        raise SystemExit(f"Unknown input scale: {input_scale}")
    return gray[None, None, ...].astype(np.float32)


def postprocess_disparity(disp: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    if mode == "none":
        return disp, mode
    if mode == "abs":
        return np.abs(disp), mode
    if mode == "negate":
        return -disp, mode
    if mode == "auto":
        finite = disp[np.isfinite(disp)]
        if finite.size and np.percentile(finite, 95) < 0:
            return -disp, "auto-negate"
        return disp, "auto-none"
    raise SystemExit(f"Unknown disparity postprocess: {mode}")


def _shape_hw(shape: list[Any] | tuple[Any, ...]) -> tuple[int | None, int | None]:
    if len(shape) >= 4 and isinstance(shape[-2], int) and isinstance(shape[-1], int):
        return int(shape[-1]), int(shape[-2])
    return None, None


def run_onnx_stereo(
    left: np.ndarray,
    right: np.ndarray,
    model_path: str,
    model_width: int | None,
    model_height: int | None,
    layout: str,
    input_scale: str,
    disp_postprocess: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit(f"onnxruntime is required for --backend onnx: {exc}") from exc

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    if not inputs:
        raise SystemExit("ONNX model has no inputs")

    inferred_w, inferred_h = _shape_hw(inputs[0].shape)
    in_w = int(model_width or inferred_w or left.shape[1])
    in_h = int(model_height or inferred_h or left.shape[0])
    size = (in_w, in_h)
    l = normalize_input(left, size, input_scale)
    r = normalize_input(right, size, input_scale)
    lg = None
    rg = None

    if layout == "auto":
        if len(inputs) >= 2:
            layout = "two-input"
        elif len(inputs) == 1:
            shape = inputs[0].shape
            channels = shape[1] if len(shape) >= 2 and isinstance(shape[1], int) else None
            batch = shape[0] if len(shape) >= 1 and isinstance(shape[0], int) else None
            if channels == 6:
                layout = "concat-channel"
            elif channels == 2:
                layout = "concat-gray-channel"
            elif batch == 2:
                layout = "concat-batch"
            else:
                layout = "concat-channel"

    feed: dict[str, np.ndarray]
    if layout == "two-input":
        if len(inputs) < 2:
            raise SystemExit("Layout two-input requires at least two ONNX inputs")
        feed = {inputs[0].name: l, inputs[1].name: r}
    elif layout == "concat-channel":
        feed = {inputs[0].name: np.concatenate([l, r], axis=1).astype(np.float32)}
    elif layout == "concat-gray-channel":
        lg = normalize_gray_input(left, size, input_scale)
        rg = normalize_gray_input(right, size, input_scale)
        feed = {inputs[0].name: np.concatenate([lg, rg], axis=1).astype(np.float32)}
    elif layout == "concat-batch":
        feed = {inputs[0].name: np.concatenate([l, r], axis=0).astype(np.float32)}
    else:
        raise SystemExit(f"Unknown ONNX layout: {layout}")

    pred = sess.run([outputs[0].name], feed)[0]
    disp = np.asarray(pred)
    disp = np.squeeze(disp)
    if disp.ndim == 3:
        # Common cases: [1,H,W] or [H,W,1]. If there are multiple channels, use first.
        if disp.shape[0] <= 4:
            disp = disp[0]
        else:
            disp = disp[..., 0]
    if disp.ndim != 2:
        raise SystemExit(f"Unsupported ONNX output shape for disparity: {pred.shape}")
    disp = disp.astype(np.float32)
    disp, used_postprocess = postprocess_disparity(disp, disp_postprocess)
    if disp.shape[:2] != left.shape[:2]:
        scale_x = left.shape[1] / float(disp.shape[1])
        disp = cv2.resize(disp, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Disparity is in pixels; if the model predicted at resized width, scale x.
        disp *= scale_x

    meta = {
        "backend": "onnx",
        "model": str(model_path),
        "layout": layout,
        "input_names": [i.name for i in inputs],
        "input_shapes": [list(i.shape) for i in inputs],
        "output_name": outputs[0].name,
        "output_shape": list(pred.shape),
        "model_size": [in_w, in_h],
        "input_scale": input_scale,
        "disparity_postprocess": used_postprocess,
    }
    return disp, meta


def run_sgbm(left: np.ndarray, right: np.ndarray, max_disp: int, block_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    num_disp = int(math.ceil(max_disp / 16.0) * 16)
    block_size = max(3, block_size | 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp, {"backend": "sgbm", "num_disparities": num_disp, "block_size": block_size}


def robust_minmax(data: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize(data: np.ndarray, invert: bool = False) -> np.ndarray:
    lo, hi = robust_minmax(data)
    norm = np.clip((data - lo) / (hi - lo), 0, 1)
    norm[~np.isfinite(norm)] = 0
    if invert:
        norm = 1.0 - norm
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~np.isfinite(data)] = (0, 0, 0)
    return color


def disparity_to_depth_mm(disp: np.ndarray, focal_px: float, baseline_mm: float) -> np.ndarray:
    depth = np.full_like(disp, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0.1)
    depth[valid] = (focal_px * baseline_mm) / disp[valid]
    return depth


def estimate_focal_px(width_px: int, hfov_deg: float) -> float:
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def summarize_map(name: str, arr: np.ndarray) -> dict[str, Any]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"name": name, "valid_ratio": 0.0}
    return {
        "name": name,
        "valid_ratio": float(finite.size / arr.size),
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def mask_summary(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    return {
        "pixels": int(mask.size),
        "face_pixels": int(np.count_nonzero(mask)),
        "face_ratio": float(np.count_nonzero(mask) / max(mask.size, 1)),
    }


def resolve_face_roi_and_masks(
    left: np.ndarray,
    right: np.ndarray,
    model_path: str,
    use_face_roi: bool,
    use_face_mask: bool,
    expand: float,
    dilate_px: int,
    roi: tuple[int, int, int, int] | None,
) -> tuple[
    tuple[int, int, int, int] | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any] | None,
]:
    if not use_face_roi and not use_face_mask:
        return roi, None, None, None

    left_pts = detect_face_landmarks_mediapipe(left, model_path)
    right_pts = detect_face_landmarks_mediapipe(right, model_path)
    h, w = left.shape[:2]
    left_face_roi = face_roi_from_landmarks(left_pts, (h, w), expand)
    right_face_roi = face_roi_from_landmarks(right_pts, (h, w), expand)

    used_roi = roi
    if use_face_roi:
        used_roi = union_rois([left_face_roi, right_face_roi], w, h)

    crop_shape = (
        (used_roi[3] - used_roi[1], used_roi[2] - used_roi[0])
        if used_roi
        else left.shape[:2]
    )
    left_mask = None
    right_mask = None
    if use_face_mask:
        left_mask = face_mask_from_landmarks(left_pts, crop_shape, used_roi, dilate_px)
        right_mask = face_mask_from_landmarks(right_pts, crop_shape, used_roi, dilate_px)

    meta = {
        "left_face_roi": list(left_face_roi),
        "right_face_roi": list(right_face_roi),
        "roi_strategy": "union-left-right" if use_face_roi else "manual-or-full",
    }
    return used_roi, left_mask, right_mask, meta


def make_preview(
    left: np.ndarray,
    right: np.ndarray,
    disp_color: np.ndarray,
    depth_color: np.ndarray | None,
    out_path: Path,
) -> None:
    h = 360

    def resize_h(img: np.ndarray) -> np.ndarray:
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)

    panels = [
        ("left", resize_h(left)),
        ("right", resize_h(right)),
        ("disparity", resize_h(disp_color)),
    ]
    if depth_color is not None:
        panels.append(("depth", resize_h(depth_color)))

    labeled = []
    for label, img in panels:
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        labeled.append(canvas)
    preview = cv2.hconcat(labeled)
    cv2.imwrite(str(out_path), preview)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimental deep stereo depth/disparity pipeline")
    ap.add_argument("--left", required=True, help="left image path")
    ap.add_argument("--right", required=True, help="right image path")
    ap.add_argument("--out", default="logs/stereo_trials/deep_stereo", help="output directory")
    ap.add_argument("--backend", choices=["onnx", "sgbm"], default="sgbm")
    ap.add_argument("--model", default=None, help="ONNX deep stereo model path")
    ap.add_argument(
        "--onnx-layout",
        choices=["auto", "two-input", "concat-channel", "concat-gray-channel", "concat-batch"],
        default="auto",
    )
    ap.add_argument("--input-scale", choices=["unit", "raw255"], default="unit")
    ap.add_argument(
        "--disparity-postprocess",
        choices=["none", "abs", "negate", "auto"],
        default="auto",
        help="post-process ONNX disparity/flow output before depth conversion",
    )
    ap.add_argument("--model-width", type=int, default=None)
    ap.add_argument("--model-height", type=int, default=None)
    ap.add_argument("--roi", default=None, help="optional x0,y0,x1,y1 crop; assumes rectified pair")
    ap.add_argument(
        "--rectify-landmarks",
        action="store_true",
        help="experimental uncalibrated rectification from face landmark correspondences",
    )
    ap.add_argument("--rectify-ransac-px", type=float, default=3.0)
    ap.add_argument("--face-roi", action="store_true", help="detect face on left image and crop same ROI on both images")
    ap.add_argument("--face-mask", action="store_true", help="mask disparity/depth to the detected face hull")
    ap.add_argument("--face-model", default="models/face_landmarker.task")
    ap.add_argument("--face-expand", type=float, default=0.25)
    ap.add_argument("--face-mask-dilate", type=int, default=12)
    ap.add_argument("--work-scale", type=float, default=1.0, help="resize cropped pair before stereo inference")
    ap.add_argument("--baseline-mm", type=float, default=100.0)
    ap.add_argument("--focal-px", type=float, default=None, help="focal length in pixels for depth conversion")
    ap.add_argument("--hfov", type=float, default=70.0, help="used to estimate focal if --focal-px is omitted")
    ap.add_argument("--max-disp", type=int, default=256, help="SGBM max disparity")
    ap.add_argument("--block-size", type=int, default=5, help="SGBM block size")
    args = ap.parse_args()

    left = read_image(args.left)
    right = read_image(args.right)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

    rectification_meta = None
    if args.rectify_landmarks:
        left, right, rectification_meta = rectify_pair_from_face_landmarks(
            left,
            right,
            args.face_model,
            args.rectify_ransac_px,
        )

    roi = parse_roi(args.roi)
    roi, face_mask, right_face_mask, face_meta = resolve_face_roi_and_masks(
        left,
        right,
        args.face_model,
        args.face_roi,
        args.face_mask,
        args.face_expand,
        args.face_mask_dilate,
        roi,
    )
    left_crop, right_crop, used_roi = crop_pair(left, right, roi)
    original_crop_shape = list(left_crop.shape)
    left_crop, right_crop, face_mask, right_face_mask = resize_work_inputs(
        left_crop,
        right_crop,
        face_mask,
        right_face_mask,
        args.work_scale,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "onnx":
        if not args.model:
            raise SystemExit("--backend onnx requires --model path")
        disp, meta = run_onnx_stereo(
            left_crop,
            right_crop,
            args.model,
            args.model_width,
            args.model_height,
            args.onnx_layout,
            args.input_scale,
            args.disparity_postprocess,
        )
    else:
        disp, meta = run_sgbm(left_crop, right_crop, args.max_disp, args.block_size)

    focal = float(args.focal_px or estimate_focal_px(left_crop.shape[1], args.hfov))
    depth = disparity_to_depth_mm(disp, focal, args.baseline_mm)

    disp_face = apply_mask_to_map(disp, face_mask)
    depth_face = apply_mask_to_map(depth, face_mask)
    disp_color = colorize(disp_face if face_mask is not None else disp)
    depth_color = colorize(depth_face if face_mask is not None else depth, invert=True)
    left_preview = apply_mask_to_bgr(left_crop, face_mask) if face_mask is not None else left_crop
    right_preview = apply_mask_to_bgr(right_crop, right_face_mask) if right_face_mask is not None else right_crop

    np.save(out_dir / "disparity.npy", disp)
    np.save(out_dir / "depth_mm.npy", depth)
    if face_mask is not None:
        np.save(out_dir / "disparity_face.npy", disp_face)
        np.save(out_dir / "depth_face_mm.npy", depth_face)
        cv2.imwrite(str(out_dir / "face_mask.png"), face_mask)
        if right_face_mask is not None:
            cv2.imwrite(str(out_dir / "right_face_mask.png"), right_face_mask)
    cv2.imwrite(str(out_dir / "left.png"), left_crop)
    cv2.imwrite(str(out_dir / "right.png"), right_crop)
    cv2.imwrite(str(out_dir / "disparity_color.png"), disp_color)
    cv2.imwrite(str(out_dir / "depth_color.png"), depth_color)
    make_preview(left_preview, right_preview, disp_color, depth_color, out_dir / "preview.png")

    report = {
        "left": str(args.left),
        "right": str(args.right),
        "roi": list(used_roi) if used_roi else None,
        "original_crop_shape": original_crop_shape,
        "work_scale": args.work_scale,
        "image_shape": list(left_crop.shape),
        "rectification": rectification_meta,
        "face": face_meta,
        "face_mask": mask_summary(face_mask),
        "right_face_mask": mask_summary(right_face_mask),
        "backend": meta,
        "baseline_mm": args.baseline_mm,
        "focal_px": focal,
        "hfov_for_focal_estimate": args.hfov if args.focal_px is None else None,
        "disparity": summarize_map("disparity_px", disp),
        "depth": summarize_map("depth_mm", depth),
        "face_disparity": summarize_map("face_disparity_px", disp_face) if face_mask is not None else None,
        "face_depth": summarize_map("face_depth_mm", depth_face) if face_mask is not None else None,
        "warning": (
            "Depth/disparity is only trustworthy for fixed, calibrated, rectified, synchronized stereo pairs."
        ),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_dir.resolve()}")
    print(json.dumps(report["disparity"], indent=2))
    print(json.dumps(report["depth"], indent=2))


if __name__ == "__main__":
    main()
