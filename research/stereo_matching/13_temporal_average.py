"""
13_temporal_average.py - Mo phong "trung binh nhieu frame" de cuu method.

Y tuong: neu nguoi dung GIU YEN trong K frame, ta lay TRUNG BINH vi tri landmark
qua K frame truoc khi do do phang. Rung ngau nhien giam -> tin hieu khoi 3D noi len.

NHUNG (phan quan trong de KHONG lap lai bay tautology):
  Nhieu landmark cua MediaPipe co HAI thanh phan:
    (1) RUNG ngau nhien  : moi frame lech mot kieu -> trung binh K frame giam theo 1/sqrt(K).
    (2) LECH co dinh      : MediaPipe dat sai mot cach NHAT QUAN theo tu the/anh sang
                            -> trung binh MAI CUNG KHONG HET. Day la TRAN (floor) that.

  Neu chi mo hinh (1) -> se ra "cang trung binh cang tot vo han" = lai ao.
  Mo hinh ca (2) -> co TRAN: separation tang roi CHUNG lai. Tran do quyet dinh
  method co dung duoc khong.

  Nhieu hieu dung sau khi trung binh K frame:
        sigma_eff(K) = sqrt( bias^2 + jitter^2 / K )
        K -> vo cuc  =>  sigma_eff -> bias   (khong the tot hon muc nay)

Chay:
    py 13_temporal_average.py
    py 13_temporal_average.py --z 350 --baseline 120 --toe 5 \
        --jitter 0.7 --bias 0.5 --kmax 30 --out figs
"""
import argparse
import importlib.util
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rig = _load("rig_sweep", "11_rig_sweep.py")


def separation_at_K(cams, face_world, flat_world, K, jitter_px, bias_px, trials, rng):
    """
    Separation sau khi trung binh K frame.
    - bias: lech co dinh, boc 1 lan cho moi 'phien', GIU NGUYEN qua K frame.
    - jitter sau trung binh K frame = N(0, jitter/sqrt(K)) (dang giai tich).
    """
    sig_jit = jitter_px / np.sqrt(K)
    rr, rf = [], []
    for _ in range(trials):
        # Lech co dinh: moi camera + moi diem co bias rieng, giu nguyen ca phien.
        biasL_f = rng.normal(0, bias_px, (len(face_world), 2))
        biasR_f = rng.normal(0, bias_px, (len(face_world), 2))
        biasL_p = rng.normal(0, bias_px, (len(flat_world), 2))
        biasR_p = rng.normal(0, bias_px, (len(flat_world), 2))

        aL = rig.project(face_world, *cams[0]) + biasL_f + rng.normal(0, sig_jit, (len(face_world), 2))
        aR = rig.project(face_world, *cams[1]) + biasR_f + rng.normal(0, sig_jit, (len(face_world), 2))
        fL = rig.project(flat_world, *cams[0]) + biasL_p + rng.normal(0, sig_jit, (len(flat_world), 2))
        fR = rig.project(flat_world, *cams[1]) + biasR_p + rng.normal(0, sig_jit, (len(flat_world), 2))

        sr = float(np.ptp(aR[:, 0])); sf = float(np.ptp(fR[:, 0]))
        r1 = rig._liveness.planarity_residual(aL, aR, scale=sr)
        r0 = rig._liveness.planarity_residual(fL, fR, scale=sf)
        if r1 and r0:
            rr.append(r1["mean_norm"]); rf.append(max(r0["mean_norm"], 1e-12))
    return float(np.median(rr) / np.median(rf))


def main():
    ap = argparse.ArgumentParser(description="Mo phong trung binh nhieu frame")
    ap.add_argument("--z", type=float, default=350.0, help="khoang cach mat (mm)")
    ap.add_argument("--baseline", type=float, default=120.0)
    ap.add_argument("--toe", type=float, default=5.0)
    ap.add_argument("--nose", type=float, default=28.0)
    ap.add_argument("--jitter", type=float, default=0.7,
                    help="thanh phan RUNG ngau nhien (px) - trung binh duoc")
    ap.add_argument("--bias", type=float, default=0.5,
                    help="thanh phan LECH co dinh (px) - trung binh KHONG het (tran)")
    ap.add_argument("--kmax", type=int, default=30, help="so frame trung binh toi da")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    face_pts, _ = rig.face_surface(args.nose)
    flat_pts, _ = rig.flat_surface()
    face_world = face_pts + np.array([0, 0, args.z])
    flat_world = flat_pts + np.array([0, 0, args.z])
    cams = rig.build_rig(2, args.baseline, args.toe)

    total_noise = np.sqrt(args.jitter ** 2 + args.bias ** 2)
    print("=== MO PHONG TRUNG BINH NHIEU FRAME ===")
    print(f"Rig: 2 cam, baseline {args.baseline:.0f}mm, toe {args.toe:.0f}do, "
          f"Z {args.z:.0f}mm, mui {args.nose:.0f}mm")
    print(f"Nhieu: rung {args.jitter}px (giam duoc) + lech co dinh {args.bias}px (tran) "
          f"= tong {total_noise:.2f}px\n")

    Ks = list(range(1, args.kmax + 1))
    seps = [separation_at_K(cams, face_world, flat_world, K,
                            args.jitter, args.bias, args.trials, rng) for K in Ks]

    print(f"{'K frame':>8} {'giay@10fps':>11} {'separation':>11}")
    for K, s in zip(Ks, seps):
        if K in (1, 3, 5, 9, 15, 25, args.kmax):
            print(f"{K:>8} {K/10:>10.1f}s {s:>10.1f}x")

    # Tran ly thuyet: K -> vo cuc, chi con lech co dinh (bias).
    sep_floor = separation_at_K(cams, face_world, flat_world, 100000,
                                args.jitter, args.bias, args.trials, rng)
    sep1 = seps[0]
    print(f"\n  Separation K=1 (1 frame)      : {sep1:.1f}x")
    print(f"  Separation tran (K -> vo cuc) : {sep_floor:.1f}x  <- gioi han that")

    # Ket luan tu dong.
    print("\n=== KET LUAN ===")
    if sep_floor >= 5:
        verdict = ("Trung binh frame CUU duoc method: tran >= 5x, du bien de tach that/gia. "
                   "Dieu kien: nguoi dung giu yen vai frame + dung gan.")
    elif sep_floor >= 3:
        verdict = ("Trung binh frame GIUP DANG KE nhung khong du manh (tran 3-5x). "
                   "Can ket hop dung that gan + landmark on dinh. Bien EER se can tinh chinh ky.")
    else:
        verdict = ("Trung binh frame KHONG cuu duoc (tran < 3x): lech co dinh cua MediaPipe "
                   "la tran that su. Method qua mong; can doi cach do (vd landmark on dinh hon, "
                   "hoac chap nhan gioi han).")
    print("  " + verdict)
    print(f"\n  Diem then chot: du trung binh BAO NHIEU frame cung khong vuot qua {sep_floor:.1f}x, "
          f"vi lech co dinh {args.bias}px khong trung binh het duoc.")

    if not args.no_plot:
        out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(Ks, seps, marker="o", label="separation thuc te")
        plt.axhline(sep_floor, color="r", ls="--", label=f"tran (K->vo cuc) = {sep_floor:.1f}x")
        plt.axhline(sep1, color="gray", ls=":", label=f"1 frame = {sep1:.1f}x")
        plt.axhline(1.0, color="k", ls="-", alpha=.3, label="1x = khong phan biet duoc")
        plt.xlabel("So frame trung binh (K)"); plt.ylabel("Separation (that/phang)")
        plt.title(f"Trung binh frame co cuu method khong?\n"
                  f"(Z={args.z:.0f}mm, jitter {args.jitter}px + bias {args.bias}px)")
        plt.legend(); plt.grid(True, alpha=.3)
        p = out_dir / "temporal_average.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"\nDa luu: {p}")


if __name__ == "__main__":
    main()
