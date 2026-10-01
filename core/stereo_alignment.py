"""
Static software alignment for the measured two-C270 stereo rig.

The current handoff metrics shift the left frame down by 14 px, the right frame
up by 14 px, then crop the clean shared region to 640x452.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class StereoAlignmentConfig:
    enabled: bool = config.STEREO_ALIGNMENT_ENABLED
    left_dx: int = config.STEREO_ALIGN_LEFT_DX
    left_dy: int = config.STEREO_ALIGN_LEFT_DY
    right_dx: int = config.STEREO_ALIGN_RIGHT_DX
    right_dy: int = config.STEREO_ALIGN_RIGHT_DY
    crop_x_start: int = config.STEREO_ALIGN_CROP_X_START
    crop_x_end: int = config.STEREO_ALIGN_CROP_X_END
    crop_y_start: int = config.STEREO_ALIGN_CROP_Y_START
    crop_y_end: int = config.STEREO_ALIGN_CROP_Y_END


DEFAULT_ALIGNMENT = StereoAlignmentConfig()


def alignment_metadata(alignment: StereoAlignmentConfig = DEFAULT_ALIGNMENT) -> dict:
    width = max(0, alignment.crop_x_end - alignment.crop_x_start)
    height = max(0, alignment.crop_y_end - alignment.crop_y_start)
    data = asdict(alignment)
    data["output_width"] = width
    data["output_height"] = height
    return data


def _clip_crop(
    frame: np.ndarray,
    alignment: StereoAlignmentConfig,
) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x0 = max(0, min(w, alignment.crop_x_start))
    x1 = max(0, min(w, alignment.crop_x_end))
    y0 = max(0, min(h, alignment.crop_y_start))
    y1 = max(0, min(h, alignment.crop_y_end))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def align_single_frame(
    frame: np.ndarray,
    camera_id: str,
    alignment: StereoAlignmentConfig = DEFAULT_ALIGNMENT,
) -> np.ndarray:
    if frame is None or not alignment.enabled:
        return frame

    if camera_id == "left":
        dx, dy = alignment.left_dx, alignment.left_dy
    elif camera_id == "right":
        dx, dy = alignment.right_dx, alignment.right_dy
    else:
        raise ValueError("camera_id must be left or right")

    h, w = frame.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(frame, matrix, (w, h))
    x0, y0, x1, y1 = _clip_crop(shifted, alignment)
    return shifted[y0:y1, x0:x1]


def align_stereo_pair(
    left: np.ndarray,
    right: np.ndarray,
    alignment: StereoAlignmentConfig = DEFAULT_ALIGNMENT,
) -> tuple[np.ndarray, np.ndarray]:
    if not alignment.enabled:
        return left, right
    return (
        align_single_frame(left, "left", alignment),
        align_single_frame(right, "right", alignment),
    )
