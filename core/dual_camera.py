"""
Two-camera demo manager for Android/IP camera rigs.

This is intentionally separate from the production Detect V4 runtime. It gives
the demo UI and research scripts a stable way to preview two network cameras
without changing the attendance scan pipeline.
"""
from __future__ import annotations

from core.camera import CameraService
import config


def _opencv_backend() -> int | None:
    if config.DUAL_CAMERA_BACKEND in {"dshow", "directshow"}:
        import cv2

        return cv2.CAP_DSHOW
    return None


class DualCameraService:
    def __init__(self):
        backend = _opencv_backend()
        self._cameras = {
            "left": CameraService(
                config.DUAL_CAMERA_LEFT_SOURCE,
                name="dual-camera-left",
                width=config.DUAL_CAMERA_WIDTH,
                height=config.DUAL_CAMERA_HEIGHT,
                fps=config.DUAL_CAMERA_FPS,
                fourcc=config.DUAL_CAMERA_FOURCC,
                backend=backend,
            ),
            "right": CameraService(
                config.DUAL_CAMERA_RIGHT_SOURCE,
                name="dual-camera-right",
                width=config.DUAL_CAMERA_WIDTH,
                height=config.DUAL_CAMERA_HEIGHT,
                fps=config.DUAL_CAMERA_FPS,
                fourcc=config.DUAL_CAMERA_FOURCC,
                backend=backend,
            ),
        }

    def start(self):
        for camera in self._cameras.values():
            camera.start()

    def stop(self):
        for camera in self._cameras.values():
            camera.stop()

    def get_camera(self, camera_id: str) -> CameraService:
        try:
            return self._cameras[camera_id]
        except KeyError as exc:
            raise KeyError(f"Unknown camera id: {camera_id}") from exc

    def get_status(self) -> dict:
        return {camera_id: camera.get_status() for camera_id, camera in self._cameras.items()}


_dual_camera: DualCameraService | None = None


def get_dual_camera() -> DualCameraService:
    global _dual_camera
    if _dual_camera is None:
        _dual_camera = DualCameraService()
    return _dual_camera
