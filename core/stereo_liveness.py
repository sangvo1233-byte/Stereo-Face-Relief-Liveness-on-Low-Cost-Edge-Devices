"""
Fail-safe stereo face liveness demo for a rigid two-C270 rig.

This module is intentionally a demo layer, not the production attendance gate.
It uses a conservative LIVE / SPOOF / INCONCLUSIVE state machine so bad frame
pairing, weak texture, blur, or missing landmarks cannot become a LIVE claim.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from core.dual_camera import DualCameraService, get_dual_camera
from core.stereo_alignment import DEFAULT_ALIGNMENT, align_stereo_pair, alignment_metadata


LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
VERDICT_LIVE = "LIVE"
VERDICT_SPOOF = "SPOOF"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

_LANDMARKER: Any | None = None
_FACE_MESH: Any | None = None
_SPOOF_SUPPRESS_UNTIL = 0.0


@dataclass(frozen=True)
class StereoThresholds:
    min_iod_px: float = config.STEREO_LIVENESS_MIN_IOD_PX
    planar_inlier_min: float = config.STEREO_LIVENESS_PLANAR_INLIER_MIN
    planar_residual_pct_max: float = config.STEREO_LIVENESS_PLANAR_RESIDUAL_PCT_MAX
    live_h_inlier_max: float = config.STEREO_LIVENESS_LIVE_H_INLIER_MAX
    live_h_residual_pct_min: float = config.STEREO_LIVENESS_LIVE_H_RESIDUAL_PCT_MIN
    f_inlier_min: float = config.STEREO_LIVENESS_F_INLIER_MIN
    f_sampson_pct_max: float = config.STEREO_LIVENESS_F_SAMPSON_PCT_MAX


DEFAULT_THRESHOLDS = StereoThresholds()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _quality(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _quality_gate(left: np.ndarray, right: np.ndarray) -> tuple[bool, dict[str, Any], list[str]]:
    lq = _quality(left)
    rq = _quality(right)
    reasons: list[str] = []
    for side, q in (("left", lq), ("right", rq)):
        if q["blur"] < config.STEREO_LIVENESS_MIN_BLUR:
            reasons.append(f"{side} blur below threshold")
        if q["brightness"] < config.STEREO_LIVENESS_MIN_BRIGHTNESS:
            reasons.append(f"{side} too dark")
        if q["brightness"] > config.STEREO_LIVENESS_MAX_BRIGHTNESS:
            reasons.append(f"{side} too bright")
    return not reasons, {"left": lq, "right": rq}, reasons


def _get_landmarker() -> Any | None:
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    model_path = Path(config.FACE_LANDMARKER_MODEL)
    if not model_path.exists():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base = mp_python.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
        _LANDMARKER = (mp, mp_vision.FaceLandmarker.create_from_options(opts))
        return _LANDMARKER
    except Exception:
        return None


def _get_face_mesh() -> Any | None:
    global _FACE_MESH
    if _FACE_MESH is not None:
        return _FACE_MESH
    try:
        import mediapipe as mp

        _FACE_MESH = (
            mp,
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            ),
        )
        return _FACE_MESH
    except Exception:
        return None


def detect_landmarks_px(image_bgr: np.ndarray) -> np.ndarray | None:
    h, w = image_bgr.shape[:2]
    task = _get_landmarker()
    if task is not None:
        mp, landmarker = task
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect(mp_img)
        if result.face_landmarks:
            return np.array(
                [(p.x * w, p.y * h) for p in result.face_landmarks[0]],
                dtype=np.float64,
            )

    mesh_state = _get_face_mesh()
    if mesh_state is None:
        return None
    _mp, mesh = mesh_state
    result = mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    return np.array(
        [(p.x * w, p.y * h) for p in result.multi_face_landmarks[0].landmark],
        dtype=np.float64,
    )


def _interocular_distance(points: np.ndarray) -> float:
    if len(points) > RIGHT_EYE_OUTER:
        return float(np.linalg.norm(points[RIGHT_EYE_OUTER] - points[LEFT_EYE_OUTER]))
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(max(x1 - x0, y1 - y0) * 0.32)


def _sampson_distance(f_mat: np.ndarray, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts_left), 1), dtype=np.float64)
    p1 = np.hstack([pts_left, ones])
    p2 = np.hstack([pts_right, ones])
    fx1 = (f_mat @ p1.T).T
    ftx2 = (f_mat.T @ p2.T).T
    num = np.sum(p2 * fx1, axis=1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def classify_landmark_pair(
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    thresholds: StereoThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    pts_left = np.asarray(pts_left, dtype=np.float64)
    pts_right = np.asarray(pts_right, dtype=np.float64)
    if pts_left.shape != pts_right.shape or pts_left.ndim != 2 or pts_left.shape[1] != 2:
        return {"verdict": VERDICT_INCONCLUSIVE, "reason": "invalid landmark shape"}
    if len(pts_left) < 12:
        return {"verdict": VERDICT_INCONCLUSIVE, "reason": "not enough landmarks"}
    if not np.isfinite(pts_left).all() or not np.isfinite(pts_right).all():
        return {"verdict": VERDICT_INCONCLUSIVE, "reason": "non-finite landmarks"}

    iod = _interocular_distance(pts_right)
    if iod < thresholds.min_iod_px:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "face too small for stereo evidence",
            "metrics": {"iod_px": iod},
        }

    ransac_px = max(1.5, 0.02 * iod)
    h_mat, h_mask = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, ransac_px)
    if h_mat is None or h_mask is None:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "homography fit failed",
            "metrics": {"iod_px": iod},
        }

    projected = cv2.perspectiveTransform(pts_left.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    h_err = np.linalg.norm(projected - pts_right, axis=1)
    h_inlier = float(h_mask.mean())
    h_residual_pct = float(h_err.mean() / iod * 100.0)
    h_p95_pct = float(np.percentile(h_err, 95) / iod * 100.0)

    f_mat, f_mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, ransac_px, 0.999)
    f_inlier = 0.0
    sampson_pct = float("inf")
    sampson_p95_pct = float("inf")
    if f_mat is not None and f_mat.shape[0] >= 3 and f_mat.shape[1] == 3:
        f_mat = f_mat[:3, :]
        f_inlier = float(f_mask.mean()) if f_mask is not None else 0.0
        sampson = _sampson_distance(f_mat, pts_left, pts_right)
        sampson_pct = float(sampson.mean() / iod * 100.0)
        sampson_p95_pct = float(np.percentile(sampson, 95) / iod * 100.0)

    metrics = _jsonable({
        "iod_px": iod,
        "ransac_px": ransac_px,
        "h_inlier": h_inlier,
        "h_residual_pct": h_residual_pct,
        "h_p95_pct": h_p95_pct,
        "f_inlier": f_inlier,
        "f_sampson_pct": sampson_pct,
        "f_sampson_p95_pct": sampson_p95_pct,
    })

    if h_inlier >= thresholds.planar_inlier_min and h_residual_pct <= thresholds.planar_residual_pct_max:
        return {
            "verdict": VERDICT_SPOOF,
            "reason": "single homography explains the face landmarks",
            "metrics": metrics,
        }

    live_geometry = (
        h_inlier <= thresholds.live_h_inlier_max
        and h_residual_pct >= thresholds.live_h_residual_pct_min
        and f_inlier >= thresholds.f_inlier_min
        and sampson_pct <= thresholds.f_sampson_pct_max
    )
    if live_geometry:
        return {
            "verdict": VERDICT_LIVE,
            "reason": "homography fails while epipolar geometry remains consistent",
            "metrics": metrics,
        }

    return {
        "verdict": VERDICT_INCONCLUSIVE,
        "reason": "geometry is ambiguous under current thresholds",
        "metrics": metrics,
    }


def analyze_frame_pair(
    left: np.ndarray,
    right: np.ndarray,
    thresholds: StereoThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    quality_ok, quality, quality_reasons = _quality_gate(left, right)
    if not quality_ok:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "; ".join(quality_reasons),
            "quality": quality,
        }

    pts_left = detect_landmarks_px(left)
    pts_right = detect_landmarks_px(right)
    if pts_left is None or pts_right is None:
        return {
            "verdict": VERDICT_INCONCLUSIVE,
            "reason": "face landmarks missing in one or both cameras",
            "quality": quality,
        }

    result = classify_landmark_pair(pts_left, pts_right, thresholds)
    result["quality"] = quality
    result["landmark_count"] = int(min(len(pts_left), len(pts_right)))
    return result


def aggregate_verdict(
    samples: list[dict[str, Any]],
    min_votes: int,
    min_live_votes: int | None = None,
) -> dict[str, Any]:
    live_votes = sum(1 for item in samples if item.get("verdict") == VERDICT_LIVE)
    spoof_votes = sum(1 for item in samples if item.get("verdict") == VERDICT_SPOOF)
    inconclusive_votes = len(samples) - live_votes - spoof_votes
    live_required = max(min_votes, min_live_votes or config.STEREO_LIVENESS_MIN_LIVE_VOTES)

    if live_votes >= live_required and spoof_votes == 0:
        verdict = VERDICT_LIVE
        reason = f"{live_votes}/{len(samples)} samples voted LIVE"
    elif spoof_votes >= min_votes and live_votes == 0:
        verdict = VERDICT_SPOOF
        reason = f"{spoof_votes}/{len(samples)} samples voted SPOOF"
    else:
        verdict = VERDICT_INCONCLUSIVE
        reason = "temporal vote did not reach a fail-safe majority"

    return {
        "verdict": verdict,
        "reason": reason,
        "votes": {
            "live": live_votes,
            "spoof": spoof_votes,
            "inconclusive": inconclusive_votes,
            "min_votes": min_votes,
            "min_live_votes": live_required,
            "total": len(samples),
        },
    }


def apply_spoof_suppression(summary: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    global _SPOOF_SUPPRESS_UNTIL

    now = time.time()
    has_spoof_evidence = summary.get("verdict") == VERDICT_SPOOF or any(
        item.get("verdict") == VERDICT_SPOOF for item in samples
    )
    if has_spoof_evidence and config.STEREO_LIVENESS_SPOOF_SUPPRESS_SECONDS > 0:
        _SPOOF_SUPPRESS_UNTIL = max(
            _SPOOF_SUPPRESS_UNTIL,
            now + config.STEREO_LIVENESS_SPOOF_SUPPRESS_SECONDS,
        )

    remaining = max(0.0, _SPOOF_SUPPRESS_UNTIL - now)
    if summary.get("verdict") == VERDICT_LIVE and remaining > 0:
        suppressed = dict(summary)
        suppressed["verdict"] = VERDICT_INCONCLUSIVE
        suppressed["reason"] = "recent planar spoof evidence suppresses LIVE"
        suppressed["spoof_suppression"] = {
            "active": True,
            "remaining_seconds": round(remaining, 2),
        }
        return suppressed

    summary["spoof_suppression"] = {
        "active": remaining > 0,
        "remaining_seconds": round(remaining, 2),
    }
    return summary


def _wait_for_frame_pair(dual: DualCameraService, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + max(0.0, timeout_seconds)
    attempts = 0
    last_status: dict[str, Any] = {}
    while True:
        attempts += 1
        left, left_ts = dual.get_camera("left").get_latest_frame_with_timestamp(copy=False)
        right, right_ts = dual.get_camera("right").get_latest_frame_with_timestamp(copy=False)
        if left is not None and right is not None and left_ts is not None and right_ts is not None:
            return {
                "ready": True,
                "attempts": attempts,
                "waited_seconds": round(max(0.0, timeout_seconds - max(0.0, deadline - time.time())), 3),
            }
        last_status = dual.get_status()
        if time.time() >= deadline:
            return {
                "ready": False,
                "attempts": attempts,
                "waited_seconds": timeout_seconds,
                "last_status": last_status,
            }
        time.sleep(0.05)


def run_stereo_liveness_check(
    dual: DualCameraService | None = None,
    sample_count: int | None = None,
    min_votes: int | None = None,
    sample_interval_seconds: float = 0.16,
    save_report: bool | None = None,
    warmup_seconds: float | None = None,
) -> dict[str, Any]:
    dual = dual or get_dual_camera()
    sample_count = sample_count or config.STEREO_LIVENESS_SAMPLE_COUNT
    min_votes = min_votes or config.STEREO_LIVENESS_MIN_VOTES
    save_report = config.STEREO_LIVENESS_LOG_JSON if save_report is None else save_report
    sample_count = max(1, int(sample_count))
    min_votes = max(1, min(int(min_votes), sample_count))
    warmup_seconds = config.STEREO_LIVENESS_WARMUP_SECONDS if warmup_seconds is None else warmup_seconds

    dual.start()
    started = time.time()
    warmup = _wait_for_frame_pair(dual, warmup_seconds)
    samples: list[dict[str, Any]] = []
    last_pair = (float("-inf"), float("-inf"))

    for index in range(sample_count):
        left, left_ts = dual.get_camera("left").get_latest_frame_with_timestamp(copy=True)
        right, right_ts = dual.get_camera("right").get_latest_frame_with_timestamp(copy=True)
        sample: dict[str, Any] = {"index": index}
        if left is None or right is None or left_ts is None or right_ts is None:
            sample.update({"verdict": VERDICT_INCONCLUSIVE, "reason": "missing frame from one or both cameras"})
        else:
            now = time.time()
            ages = (now - left_ts, now - right_ts)
            host_skew_ms = abs(left_ts - right_ts) * 1000.0
            sample["host_frame_delta_ms"] = host_skew_ms
            sample["host_frame_age_seconds"] = {"left": ages[0], "right": ages[1]}
            if not all(0 <= age <= config.STEREO_LIVENESS_MAX_FRAME_AGE_SECONDS for age in ages):
                sample.update({"verdict": VERDICT_INCONCLUSIVE, "reason": "stale or invalid frame timestamp"})
            elif left_ts <= last_pair[0] or right_ts <= last_pair[1]:
                sample.update({"verdict": VERDICT_INCONCLUSIVE, "reason": "frame pair has not advanced"})
            elif host_skew_ms > config.STEREO_LIVENESS_MAX_HOST_SKEW_MS:
                sample.update({
                    "verdict": VERDICT_INCONCLUSIVE,
                    "reason": "host frame timestamp delta too high",
                })
            else:
                left, right = align_stereo_pair(left, right)
                sample["aligned_shape"] = {
                    "left": list(left.shape[:2]),
                    "right": list(right.shape[:2]),
                }
                sample.update(analyze_frame_pair(left, right))
            last_pair = (max(last_pair[0], left_ts), max(last_pair[1], right_ts))
        samples.append(_jsonable(sample))
        if index < sample_count - 1:
            time.sleep(sample_interval_seconds)

    summary = apply_spoof_suppression(aggregate_verdict(samples, min_votes=min_votes), samples)
    report = {
        "success": True,
        "mode": "two_c270_stereo_demo",
        "claim_boundary": (
            "Demo evidence only: C270 frames are not hardware-synchronized; "
            "LIVE requires clean multi-frame geometry, otherwise return INCONCLUSIVE."
        ),
        "summary": summary,
        "samples": samples,
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "alignment": alignment_metadata(DEFAULT_ALIGNMENT),
        "warmup": warmup,
        "camera_status": dual.get_status(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if save_report:
        report["report_path"] = str(_save_report(report))
    return report


def _save_report(report: dict[str, Any]) -> Path:
    out_dir = config.LOGS_DIR / "stereo_liveness"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stereo_liveness_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(_jsonable(report), f, ensure_ascii=False, indent=2)
    return path
