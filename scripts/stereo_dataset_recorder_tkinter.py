"""
Tkinter recorder for a local stereo-liveness pilot dataset.

Records paired, aligned C270 clips under datasets/stereo_liveness/. This stores
video frames, so keep the output local and never commit it.
"""
from __future__ import annotations

import argparse
import json
import time
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
import sys
from tkinter import ttk

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.stereo_alignment import DEFAULT_ALIGNMENT, align_stereo_pair, alignment_metadata


LABELS = [
    "LIVE_USER",
    "SCREEN_SPOOF",
    "PRINT_SPOOF",
    "EMPTY",
    "PARTIAL",
]

LABEL_DURATIONS_SECONDS = {
    "LIVE_USER": 120,
    "SCREEN_SPOOF": 120,
    "PRINT_SPOOF": 60,
    "EMPTY": 60,
    "PARTIAL": 60,
}


def _camera_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _backend(value: str) -> int | None:
    if value.lower() in {"dshow", "directshow"}:
        return cv2.CAP_DSHOW
    return None


def _open_capture(source, backend: int | None, width: int, height: int, fps: int, fourcc: str):
    cap = cv2.VideoCapture(source, backend) if backend is not None and isinstance(source, int) else cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        if fourcc and len(fourcc) >= 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _photo_image(frame_bgr: np.ndarray, max_width: int = 1180) -> tk.PhotoImage:
    frame = frame_bgr
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".ppm", rgb)
    if not ok:
        raise RuntimeError("Failed to encode preview frame")
    return tk.PhotoImage(data=encoded.tobytes(), format="PPM")


class StereoDatasetRecorder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Stereo Dataset Recorder")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        backend = _backend(args.backend)
        self.left_cap = _open_capture(_camera_source(args.left), backend, args.width, args.height, args.fps, args.fourcc)
        self.right_cap = _open_capture(_camera_source(args.right), backend, args.width, args.height, args.fps, args.fourcc)

        self.recording = False
        self.record_started_at = 0.0
        self.record_until = 0.0
        self.frame_count = 0
        self.clip_dir: Path | None = None
        self.left_writer = None
        self.right_writer = None
        self.preview_image = None
        self.last_left = None
        self.last_right = None

        self.label_var = tk.StringVar(value=LABELS[0])
        self.duration_var = tk.StringVar(value=str(LABEL_DURATIONS_SECONDS.get(LABELS[0], args.duration)))
        self.session_var = tk.StringVar(value=args.session or time.strftime("%Y%m%d_%H%M%S"))
        self.status_var = tk.StringVar(value="idle")

        self._build_ui()
        self._tick()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Label").pack(side=tk.LEFT)
        label_box = ttk.Combobox(top, textvariable=self.label_var, values=LABELS, width=16, state="readonly")
        label_box.pack(side=tk.LEFT, padx=6)
        label_box.bind("<<ComboboxSelected>>", self._apply_label_duration)

        ttk.Label(top, text="Duration seconds").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.duration_var, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Session").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.session_var, width=18).pack(side=tk.LEFT, padx=6)

        ttk.Button(top, text="Record", command=self.start_recording).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Stop", command=self.stop_recording).pack(side=tk.LEFT)

        self.preview = ttk.Label(self.root)
        self.preview.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        ttk.Label(self.root, textvariable=self.status_var, padding=10).pack(fill=tk.X)

    def _apply_label_duration(self, _event=None):
        duration = LABEL_DURATIONS_SECONDS.get(self.label_var.get())
        if duration is not None and not self.recording:
            self.duration_var.set(str(duration))
            self.status_var.set(f"preset {self.label_var.get()} duration={duration}s")

    def start_recording(self):
        if self.recording:
            return
        try:
            duration = max(1.0, float(self.duration_var.get()))
        except ValueError:
            duration = float(self.args.duration)

        label = self.label_var.get()
        clip_id = time.strftime("%Y%m%d_%H%M%S")
        self.clip_dir = Path(self.args.output) / self.session_var.get() / label / clip_id
        self.clip_dir.mkdir(parents=True, exist_ok=True)

        out_size = (
            DEFAULT_ALIGNMENT.crop_x_end - DEFAULT_ALIGNMENT.crop_x_start,
            DEFAULT_ALIGNMENT.crop_y_end - DEFAULT_ALIGNMENT.crop_y_start,
        )
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.left_writer = cv2.VideoWriter(str(self.clip_dir / "left.avi"), fourcc, self.args.fps, out_size)
        self.right_writer = cv2.VideoWriter(str(self.clip_dir / "right.avi"), fourcc, self.args.fps, out_size)
        if not self.left_writer.isOpened() or not self.right_writer.isOpened():
            self.stop_recording()
            self.status_var.set("writer open failed")
            return

        metadata = {
            "label": label,
            "session": self.session_var.get(),
            "clip_id": clip_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": duration,
            "fps": self.args.fps,
            "left_source": self.args.left,
            "right_source": self.args.right,
            "width": self.args.width,
            "height": self.args.height,
            "fourcc": self.args.fourcc,
            "backend": self.args.backend,
            "privacy": "contains local video frames; do not commit or share unless intentionally anonymized",
            "alignment": alignment_metadata(),
            "alignment_config": asdict(DEFAULT_ALIGNMENT),
        }
        (self.clip_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.recording = True
        self.record_started_at = time.time()
        self.record_until = self.record_started_at + duration
        self.frame_count = 0
        self.status_var.set(f"recording {label} -> {self.clip_dir}")

    def stop_recording(self):
        self.recording = False
        if self.left_writer is not None:
            self.left_writer.release()
            self.left_writer = None
        if self.right_writer is not None:
            self.right_writer.release()
            self.right_writer = None
        if self.clip_dir is not None:
            stats = {
                "frames": self.frame_count,
                "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_seconds": round(max(0.0, time.time() - self.record_started_at), 3),
            }
            (self.clip_dir / "recording_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        self.status_var.set(f"stopped, frames={self.frame_count}")

    def _read_pair(self):
        ok_l, left = self.left_cap.read()
        ok_r, right = self.right_cap.read()
        if not ok_l or not ok_r:
            return None, None
        return align_stereo_pair(left, right)

    def _tick(self):
        left, right = self._read_pair()
        if left is not None and right is not None:
            self.last_left = left
            self.last_right = right
            if self.recording and self.left_writer is not None and self.right_writer is not None:
                self.left_writer.write(left)
                self.right_writer.write(right)
                self.frame_count += 1
                remaining = max(0.0, self.record_until - time.time())
                self.status_var.set(f"recording {self.label_var.get()} | frames={self.frame_count} | remaining={remaining:.1f}s")
                if time.time() >= self.record_until:
                    self.stop_recording()

            preview = np.hstack([left, right])
            self.preview_image = _photo_image(preview)
            self.preview.configure(image=self.preview_image)
        else:
            self.status_var.set("waiting for both cameras")

        self.root.after(max(1, int(1000 / max(1, self.args.fps))), self._tick)

    def close(self):
        self.stop_recording()
        self.left_cap.release()
        self.right_cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Record aligned stereo clips with manual labels.")
    parser.add_argument("--left", default=str(config.DUAL_CAMERA_LEFT_SOURCE))
    parser.add_argument("--right", default=str(config.DUAL_CAMERA_RIGHT_SOURCE))
    parser.add_argument("--backend", default=config.DUAL_CAMERA_BACKEND)
    parser.add_argument("--width", type=int, default=config.DUAL_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=config.DUAL_CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=config.DUAL_CAMERA_FPS)
    parser.add_argument("--fourcc", default=config.DUAL_CAMERA_FOURCC)
    parser.add_argument("--duration", type=float, default=60.0, help="Fallback duration when a label has no preset.")
    parser.add_argument("--session", default="")
    parser.add_argument("--output", default="datasets/stereo_liveness")
    return parser.parse_args()


def main():
    StereoDatasetRecorder(parse_args()).run()


if __name__ == "__main__":
    main()
