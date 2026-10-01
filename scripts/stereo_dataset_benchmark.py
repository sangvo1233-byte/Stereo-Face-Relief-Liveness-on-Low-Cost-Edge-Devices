"""
Offline benchmark for locally recorded stereo liveness clips.

Reads datasets created by stereo_dataset_recorder_tkinter.py and writes numeric
metrics only. It does not export frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from pathlib import Path
import sys
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core.stereo_liveness import (
    VERDICT_INCONCLUSIVE,
    VERDICT_LIVE,
    VERDICT_SPOOF,
    aggregate_verdict,
    analyze_frame_pair,
)
from core.stereo_alignment import alignment_metadata


SPOOF_LABELS = {"SCREEN_SPOOF", "PHONE_SPOOF", "PRINT_SPOOF", "LAPTOP_SPOOF"}
LIVE_LABELS = {"LIVE_USER"}
NON_FACE_LABELS = {"EMPTY", "PARTIAL"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def discover_clips(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("metadata.json"))


def load_metadata(clip_dir: Path) -> dict[str, Any]:
    return json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))


def is_current_alignment(meta: dict[str, Any]) -> bool:
    expected = alignment_metadata()
    actual = meta.get("alignment") or {}
    keys = [
        "left_dx",
        "left_dy",
        "right_dx",
        "right_dy",
        "crop_x_start",
        "crop_x_end",
        "crop_y_start",
        "crop_y_end",
        "output_width",
        "output_height",
    ]
    return all(actual.get(key) == expected.get(key) for key in keys)


def row_from_result(clip_id: str, label: str, frame_index: int, result: dict[str, Any], summary: dict[str, Any] | None):
    quality = result.get("quality") or {}
    metrics = result.get("metrics") or {}
    votes = (summary or {}).get("votes") or {}
    return {
        "clip_id": clip_id,
        "label": label,
        "frame_index": frame_index,
        "sample_verdict": result.get("verdict", VERDICT_INCONCLUSIVE),
        "window_verdict": (summary or {}).get("verdict", ""),
        "reason": (summary or {}).get("reason") or result.get("reason", ""),
        "landmark_count": result.get("landmark_count", ""),
        "left_brightness": (quality.get("left") or {}).get("brightness", ""),
        "right_brightness": (quality.get("right") or {}).get("brightness", ""),
        "left_blur": (quality.get("left") or {}).get("blur", ""),
        "right_blur": (quality.get("right") or {}).get("blur", ""),
        "iod_px": metrics.get("iod_px", ""),
        "h_inlier": metrics.get("h_inlier", ""),
        "h_residual_pct": metrics.get("h_residual_pct", ""),
        "h_p95_pct": metrics.get("h_p95_pct", ""),
        "f_inlier": metrics.get("f_inlier", ""),
        "f_sampson_pct": metrics.get("f_sampson_pct", ""),
        "live_votes": votes.get("live", ""),
        "spoof_votes": votes.get("spoof", ""),
        "inconclusive_votes": votes.get("inconclusive", ""),
    }


def evaluate_clip(clip_dir: Path, stride: int, window: int, max_frames: int | None) -> list[dict[str, Any]]:
    meta = load_metadata(clip_dir)
    label = meta["label"]
    clip_id = f"{meta.get('session', '')}/{label}/{meta.get('clip_id', clip_dir.name)}"
    left_cap = cv2.VideoCapture(str(clip_dir / "left.avi"))
    right_cap = cv2.VideoCapture(str(clip_dir / "right.avi"))
    rows: list[dict[str, Any]] = []
    recent = deque(maxlen=max(1, window))
    frame_index = -1

    while True:
        ok_l, left = left_cap.read()
        ok_r, right = right_cap.read()
        if not ok_l or not ok_r:
            break
        frame_index += 1
        if max_frames is not None and frame_index >= max_frames:
            break
        if frame_index % max(1, stride) != 0:
            continue

        result = _jsonable(analyze_frame_pair(left, right))
        recent.append(result)
        summary = None
        if len(recent) == recent.maxlen:
            summary = aggregate_verdict(list(recent), min_votes=len(recent), min_live_votes=config.STEREO_LIVENESS_MIN_LIVE_VOTES)
        rows.append(row_from_result(clip_id, label, frame_index, result, summary))

    left_cap.release()
    right_cap.release()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable_rows = [row for row in rows if row["window_verdict"]]
    verdict_key = "window_verdict"
    counts = Counter((row["label"], row[verdict_key]) for row in usable_rows)
    labels = sorted({row["label"] for row in usable_rows})
    verdicts = [VERDICT_LIVE, VERDICT_SPOOF, VERDICT_INCONCLUSIVE]
    matrix = {
        label: {verdict: counts.get((label, verdict), 0) for verdict in verdicts}
        for label in labels
    }

    spoof_total = sum(sum(matrix.get(label, {}).values()) for label in SPOOF_LABELS)
    spoof_live = sum(matrix.get(label, {}).get(VERDICT_LIVE, 0) for label in SPOOF_LABELS)
    live_total = sum(sum(matrix.get(label, {}).values()) for label in LIVE_LABELS)
    live_reject = sum(
        matrix.get(label, {}).get(VERDICT_SPOOF, 0) + matrix.get(label, {}).get(VERDICT_INCONCLUSIVE, 0)
        for label in LIVE_LABELS
    )
    apcer = spoof_live / spoof_total if spoof_total else None
    bpcer = live_reject / live_total if live_total else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_count": len(rows),
        "window_count": len(usable_rows),
        "confusion_matrix": matrix,
        "pilot_metrics": {
            "apcer_spoof_to_live": apcer,
            "bpcer_live_rejected": bpcer,
            "acer": acer,
            "notes": "Pilot metric only. INCONCLUSIVE is treated as reject for BPCER and non-live for spoof attacks.",
        },
        "privacy": "summary contains numeric metrics only; source clips remain local under datasets/ and are gitignored",
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "samples.jsonl"
    csv_path = output_dir / "samples.csv"
    summary_path = output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "csv": str(csv_path), "summary": str(summary_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark recorded stereo clips.")
    parser.add_argument("--dataset", default="datasets/stereo_liveness")
    parser.add_argument("--output", default="output/stereo_eval")
    parser.add_argument("--stride", type=int, default=5, help="Evaluate every Nth frame.")
    parser.add_argument("--window", type=int, default=3, help="Temporal verdict window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap per clip; 0 means no cap.")
    parser.add_argument("--allow-mixed-alignment", action="store_true", help="Include clips recorded with older alignment.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset)
    clips = discover_clips(root)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for clip_dir in clips:
        meta = load_metadata(clip_dir)
        if not args.allow_mixed_alignment and not is_current_alignment(meta):
            skipped.append(str(clip_dir))
            continue
        rows.extend(evaluate_clip(clip_dir, args.stride, args.window, args.max_frames or None))
    summary = summarize(rows)
    summary["skipped_clips"] = skipped
    paths = write_outputs(rows, summary, Path(args.output) / time.strftime("%Y%m%d_%H%M%S"))
    print(json.dumps({"clips": len(clips), "rows": len(rows), "outputs": paths, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
