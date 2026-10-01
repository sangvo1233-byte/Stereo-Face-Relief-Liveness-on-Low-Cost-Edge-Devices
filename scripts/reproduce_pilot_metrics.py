"""Recount public pilot CSV windows and verify their archived JSON summaries."""
import csv
import json
import math
from collections import Counter
from pathlib import Path


def summarize(rows):
    counts = Counter()
    for row in rows:
        label, verdict = row["label"], row["window_verdict"]
        if label not in {"LIVE_USER", "SCREEN_SPOOF", "EMPTY", "PARTIAL"}:
            raise ValueError(f"Unexpected pilot label: {label}")
        if verdict not in {"", "LIVE", "SPOOF", "INCONCLUSIVE"}:
            raise ValueError(f"Unexpected verdict: {verdict}")
        if verdict:
            counts[label, verdict] += 1
    matrix = {
        label: {v: counts[label, v] for v in ("LIVE", "SPOOF", "INCONCLUSIVE")}
        for label in sorted({label for label, _ in counts})
    }
    live_total = sum(matrix.get("LIVE_USER", {}).values())
    spoof_total = sum(matrix.get("SCREEN_SPOOF", {}).values())
    if not live_total or not spoof_total:
        raise ValueError("Pilot metrics need both bona fide and attack windows")
    apcer = counts["SCREEN_SPOOF", "LIVE"] / spoof_total
    bpcer = (live_total - counts["LIVE_USER", "LIVE"]) / live_total
    return {"sample_count": len(rows), "window_count": sum(counts.values()),
            "confusion_matrix": matrix,
            "pilot_metrics": {"apcer_spoof_to_live": apcer,
                              "bpcer_live_rejected": bpcer, "acer": (apcer + bpcer) / 2}}


def main():
    root = Path(__file__).resolve().parents[1] / "results" / "pilot"
    combined = []
    runs = ("20260714_160804", "20260718_001027")
    for run in runs:
        with (root / run / "samples.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        expected = json.loads((root / run / "summary.json").read_text(encoding="utf-8"))
        actual = summarize(rows)
        for key in ("sample_count", "window_count", "confusion_matrix"):
            if actual[key] != expected[key]:
                raise ValueError(f"{run}: archived {key} does not match CSV")
        for key, value in actual["pilot_metrics"].items():
            if not math.isclose(value, expected["pilot_metrics"][key], abs_tol=1e-12):
                raise ValueError(f"{run}: archived {key} does not match CSV")
        combined.extend(rows)
        print(run, json.dumps(actual, sort_keys=True))
    result = summarize(combined)
    if result["window_count"] != 903 or not math.isclose(result["pilot_metrics"]["bpcer_live_rejected"], 33 / 217):
        raise ValueError("Combined counts no longer match the published pilot summary")
    print("combined", json.dumps(result, sort_keys=True))
    print("PASS: archived CSV counts and pilot metrics match their summaries.")


if __name__ == "__main__":
    main()
