"""
Dual camera and stereo liveness demo routes.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.dual_camera import get_dual_camera
from core.stereo_alignment import align_single_frame, align_stereo_pair, alignment_metadata
from core.stereo_liveness import run_stereo_liveness_check

router = APIRouter(tags=["dual-camera"])

_VALID_IDS = {"left", "right"}
_BLANK_FRAMES: dict[str, bytes] = {}


def _blank_frame(label: str) -> bytes:
    if label not in _BLANK_FRAMES:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            img,
            f"No {label} camera",
            (145, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
        )
        cv2.putText(
            img,
            "Check camera source / USB index",
            (132, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 60, 60),
            1,
        )
        _, buf = cv2.imencode(".jpg", img)
        _BLANK_FRAMES[label] = buf.tobytes()
    return _BLANK_FRAMES[label]


def _encode_frame(frame: np.ndarray | None, label: str) -> bytes:
    if frame is None:
        return _blank_frame(label)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else _blank_frame(label)


def _camera_or_404(camera_id: str):
    if camera_id not in _VALID_IDS:
        raise HTTPException(status_code=404, detail="camera_id must be left or right")
    return get_dual_camera().get_camera(camera_id)


def _generate_single(camera_id: str):
    cam = _camera_or_404(camera_id)
    cam.start()
    while True:
        frame = cam.get_latest_frame(copy=True)
        if frame is not None:
            frame = align_single_frame(frame, camera_id)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + _encode_frame(frame, camera_id)
            + b"\r\n"
        )
        time.sleep(1.0 / 20)


def _normalize_height(frame: np.ndarray, target_h: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame
    target_w = max(1, int(w * target_h / h))
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    return out


def _generate_grid():
    dual = get_dual_camera()
    dual.start()
    while True:
        left = dual.get_camera("left").get_latest_frame(copy=True)
        right = dual.get_camera("right").get_latest_frame(copy=True)
        if left is not None and right is not None:
            left, right = align_stereo_pair(left, right)

        left_img = cv2.imdecode(np.frombuffer(_encode_frame(left, "left"), np.uint8), cv2.IMREAD_COLOR)
        right_img = cv2.imdecode(np.frombuffer(_encode_frame(right, "right"), np.uint8), cv2.IMREAD_COLOR)
        left_img = _label(_normalize_height(left_img), "LEFT")
        right_img = _label(_normalize_height(right_img), "RIGHT")

        grid = cv2.hconcat([left_img, right_img])
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 72])
        payload = buf.tobytes() if ok else _blank_frame("dual")
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        time.sleep(1.0 / 20)


@router.post("/api/dual-camera/start")
async def start_dual_camera():
    dual = get_dual_camera()
    dual.start()
    return {"success": True, "status": dual.get_status()}


@router.post("/api/dual-camera/stop")
async def stop_dual_camera():
    dual = get_dual_camera()
    dual.stop()
    return {"success": True, "status": dual.get_status()}


@router.get("/api/dual-camera/status")
async def dual_camera_status():
    return get_dual_camera().get_status()


@router.get("/api/stereo-liveness/status")
async def stereo_liveness_status():
    return {
        "success": True,
        "mode": "two_c270_stereo_demo",
        "claim_boundary": "Demo evidence only; no hardware sync on Logitech C270, fail-safe INCONCLUSIVE is expected.",
        "alignment": alignment_metadata(),
        "camera_status": get_dual_camera().get_status(),
    }


@router.post("/api/stereo-liveness/check")
async def stereo_liveness_check(
    sample_count: int | None = None,
    save_report: bool | None = None,
    warmup_seconds: float | None = None,
):
    return run_stereo_liveness_check(
        sample_count=sample_count,
        save_report=save_report,
        warmup_seconds=warmup_seconds,
    )


@router.get("/api/live/dual/{camera_id}")
async def dual_camera_stream(camera_id: str):
    _camera_or_404(camera_id)
    return StreamingResponse(
        _generate_single(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/api/live/dual-grid")
async def dual_camera_grid_stream():
    return StreamingResponse(
        _generate_grid(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
