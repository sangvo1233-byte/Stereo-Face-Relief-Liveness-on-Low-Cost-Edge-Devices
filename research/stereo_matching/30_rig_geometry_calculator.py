"""
30_rig_geometry_calculator.py - quick geometry calculator for a 2-camera rig.

This is a design aid, not a verification result. It estimates:
  - focal length in pixels from image width + horizontal FOV
  - symmetric toe-in angle so both cameras look at the face center
  - disparity at working distance
  - expected disparity change caused by face relief (nose/cheek depth)
  - depth uncertainty from correspondence noise

Run:
    py research/stereo_matching/30_rig_geometry_calculator.py
    py research/stereo_matching/30_rig_geometry_calculator.py --z 500 --baselines 60 80 100 120
"""
from __future__ import annotations

import argparse
import math


def focal_px(width_px: float, hfov_deg: float) -> float:
    """Convert horizontal FOV to focal length in pixels."""
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def toe_in_each_camera_deg(baseline_mm: float, z_mm: float) -> float:
    """Symmetric rig: cameras at +/-B/2, both looking at target center (0, Z)."""
    return math.degrees(math.atan2(baseline_mm / 2.0, z_mm))


def disparity_px(f_px: float, baseline_mm: float, z_mm: float) -> float:
    """Approximate rectified disparity at depth Z."""
    return f_px * baseline_mm / z_mm


def relief_signal_px(f_px: float, baseline_mm: float, z_mm: float, relief_mm: float) -> float:
    """Disparity delta between face plane at Z and a protruding point at Z-relief."""
    z_near = max(1.0, z_mm - relief_mm)
    return abs(f_px * baseline_mm * (1.0 / z_near - 1.0 / z_mm))


def depth_sigma_mm(f_px: float, baseline_mm: float, z_mm: float, disparity_noise_px: float) -> float:
    """First-order stereo depth uncertainty: dZ ~= Z^2/(fB) * dd."""
    return (z_mm * z_mm / max(f_px * baseline_mm, 1e-9)) * disparity_noise_px


def required_baseline_mm(
    f_px: float,
    z_mm: float,
    relief_mm: float,
    disparity_noise_px: float,
    min_snr: float,
) -> float:
    """Baseline needed for relief disparity signal >= min_snr * noise."""
    z_near = max(1.0, z_mm - relief_mm)
    denom = f_px * abs(1.0 / z_near - 1.0 / z_mm)
    return (min_snr * disparity_noise_px) / max(denom, 1e-9)


def classify_candidate(b_over_z: float, toe_deg: float, snr: float) -> str:
    """Soft design label. This is not a real-camera verdict."""
    notes = []
    if b_over_z < 0.10:
        notes.append("weak B/Z")
    elif b_over_z < 0.18:
        notes.append("dense-only candidate")
    elif b_over_z < 0.35:
        notes.append("good dense candidate")
    else:
        notes.append("large baseline")

    if toe_deg > 10.0:
        notes.append("toe high")
    if snr < 4.0:
        notes.append("low relief SNR")
    elif snr >= 8.0:
        notes.append("strong relief SNR")
    return ", ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description="2-camera rig geometry calculator")
    ap.add_argument("--width", type=float, default=1280.0, help="image width in pixels")
    ap.add_argument("--hfov", type=float, default=70.0, help="horizontal FOV in degrees")
    ap.add_argument("--z", type=float, default=500.0, help="working distance to face center in mm")
    ap.add_argument("--z-min", type=float, default=400.0, help="nearest expected face distance in mm")
    ap.add_argument("--z-max", type=float, default=600.0, help="farthest expected face distance in mm")
    ap.add_argument("--relief", type=float, default=25.0, help="face relief to detect in mm")
    ap.add_argument(
        "--noise",
        type=float,
        default=0.5,
        help="correspondence disparity noise in px; use 0.2-0.5 dense, 1-3 sparse",
    )
    ap.add_argument("--snr", type=float, default=6.0, help="minimum relief signal/noise target")
    ap.add_argument(
        "--baselines",
        type=float,
        nargs="+",
        default=[60.0, 80.0, 100.0, 120.0, 150.0],
        help="candidate baselines in mm",
    )
    args = ap.parse_args()

    f_px = focal_px(args.width, args.hfov)
    req = required_baseline_mm(f_px, args.z, args.relief, args.noise, args.snr)

    print("2-CAMERA RIG GEOMETRY CALCULATOR")
    print("Status: design estimate, not hardware verification")
    print(f"image width={args.width:.0f}px  HFOV={args.hfov:.1f}deg  focal~{f_px:.1f}px")
    print(f"working Z={args.z:.0f}mm  range={args.z_min:.0f}-{args.z_max:.0f}mm  relief={args.relief:.0f}mm")
    print(f"correspondence noise={args.noise:.2f}px  target SNR={args.snr:.1f}")
    print(f"minimum baseline for target relief SNR at Z={args.z:.0f}mm: {req:.1f}mm")
    print()
    print(
        f"{'B(mm)':>6} {'B/Z':>6} {'toe/cam':>8} {'conv':>7} "
        f"{'disp@Z':>8} {'relief':>8} {'SNR':>6} {'dZ@Z':>7} {'notes':<28}"
    )
    print("-" * 100)

    for baseline in args.baselines:
        bz = baseline / args.z
        toe = toe_in_each_camera_deg(baseline, args.z)
        conv = toe * 2.0
        disp = disparity_px(f_px, baseline, args.z)
        relief = relief_signal_px(f_px, baseline, args.z, args.relief)
        snr = relief / max(args.noise, 1e-9)
        dz = depth_sigma_mm(f_px, baseline, args.z, args.noise)
        notes = classify_candidate(bz, toe, snr)
        print(
            f"{baseline:6.0f} {bz:6.3f} {toe:8.2f} {conv:7.2f} "
            f"{disp:8.1f} {relief:8.2f} {snr:6.1f} {dz:7.2f} {notes:<28}"
        )

    print()
    print("Guidance:")
    print("- Use toe/cam ~= atan((B/2)/Z_target); this centers both cameras on the face.")
    print("- Depth/relief signal mainly comes from B/Z and correspondence quality, not toe-in.")
    print("- Start pilot with B=80-120mm at Z=450-550mm; prefer dense/sub-pixel matching.")
    print("- Sparse landmarks may need larger B/Z and can false-reject real faces.")


if __name__ == "__main__":
    main()
