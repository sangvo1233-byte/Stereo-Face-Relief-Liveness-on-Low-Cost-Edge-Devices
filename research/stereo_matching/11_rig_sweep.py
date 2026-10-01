"""
Cong cu SWEEP hinh hoc: chon RIG toi uu (so cam N, baseline B, goc hoi tu theta).

Muc dich: TRUOC KHI dong rig phan cung + thu data that, tra loi bang SO:
  - Baseline B bao nhieu mm thi tach biet that/phang manh nhat?
  - Goc hoi tu (toe-in) theta bao nhieu do?
  - 2 hay 3 cam? Cam thu 3 co dang tien khong?
  - Cau hinh do on dinh the nao khi nguoi dung dung o cac khoang cach Z khac nhau?

Phuong phap (MUC A - luoi thu tuc, KHONG can model/render/webcam):
  - Tao mat 3D thu tuc (mui nho ra) giong 04_synthetic_test.py.
  - Dat N camera tren mot rig (offset ngang +/- , toe-in theta), chieu landmark
    xuong tung view bang pinhole (cv2.projectPoints).
  - THEM NHIEU landmark thuc te (Gaussian pixel) de PHA tautology: khong con
    tach biet vo han nhu 04, ma co "san nhieu" giong MediaPipe that.
  - Chay DUNG ham loi planarity_residual cua 03 tren tung cap view.
  - Do 3 dai luong cho moi cau hinh:
        separation = residual_that / residual_phang   (cang cao cang de phan biet)
        visibility = ti le landmark "dang tin" o CA 2 view (chong self-occlusion)
        CV         = do bien thien residual qua nhieu lan thu (cang thap cang on)

GIOI HAN (ghi ro trong bao cao): day la NGHIEM HINH HOC LY TUONG. Nhieu landmark
that cua MediaPipe + do cong giay/bezel man hinh se dich diem toi uu. Ket qua sweep
la DIEM KHOI DAU THIET KE rig, phai hieu chinh lai bang pilot that.

Chay:
    py 11_rig_sweep.py
    py 11_rig_sweep.py --z 450 600 750 --baselines 40 60 80 100 120 140 160 \
                       --angles 0 5 10 15 20 --noise-px 1.0 --trials 24 --out figs
"""

import argparse
import math
import pathlib

import numpy as np

# --- Tai ham loi planarity_residual THAT tu 03 (test dung code that, khong copy) ---
import importlib.util


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        mod_name, pathlib.Path(__file__).with_name(file_name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_liveness = _load("liveness_planar", "03_liveness_planar.py")

# Camera noi tai gia lap (giong 04): webcam ~640x480, tieu cu 700px -> FOV ~49 do.
IMG_W, IMG_H = 640, 480
K = np.array([[700.0, 0, IMG_W / 2.0],
              [0, 700.0, IMG_H / 2.0],
              [0, 0, 1]], dtype=np.float64)


# ------------------------------------------------------------------ mat 3D + normal
def face_surface(nose_depth_mm=28.0, nx=9, ny=11):
    """
    Luoi diem kieu khuon mat + normal be mat (huong RA phia camera).
    nose_depth_mm: do noi khoi (mui nho ra bao nhieu mm so voi ria mat) = delta-Z.
    Tra ve (pts (M,3), normals (M,3)) trong he toa do mat (goc o tam, mat huong -Z).
    """
    xs = np.linspace(-50, 50, nx)
    ys = np.linspace(-65, 65, ny)
    sx, sy = 35.0, 45.0
    pts, nrm = [], []
    for x in xs:
        for y in ys:
            e = math.exp(-((x / sx) ** 2 + (y / sy) ** 2))
            z = -nose_depth_mm * e                     # nho ra phia camera (-Z)
            # Gradient cua z -> normal. z = -d*exp(-(x^2/sx^2 + y^2/sy^2))
            dzdx = -nose_depth_mm * e * (-2 * x / sx ** 2)
            dzdy = -nose_depth_mm * e * (-2 * y / sy ** 2)
            n = np.array([dzdx, dzdy, -1.0])           # huong ve -Z (phia camera)
            n /= np.linalg.norm(n)
            pts.append((x, y, z))
            nrm.append(n)
    return np.array(pts, dtype=np.float64), np.array(nrm, dtype=np.float64)


def flat_surface(nx=9, ny=11):
    """Mat PHANG (anh in/man hinh): cung luoi x,y nhung z=0, normal = [0,0,-1]."""
    xs = np.linspace(-50, 50, nx)
    ys = np.linspace(-65, 65, ny)
    pts = np.array([(x, y, 0.0) for x in xs for y in ys], dtype=np.float64)
    nrm = np.tile(np.array([0.0, 0.0, -1.0]), (len(pts), 1))
    return pts, nrm


# ------------------------------------------------------------------ camera
def yaw_R(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)],
                     [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]], dtype=np.float64)


def make_camera(x_off_mm, toe_in_deg):
    """
    Camera tren rig: tam tai (x_off, 0, 0), xoay toe-in HUONG VAO tam.
    Cam ben trai (x_off<0) toe phai, cam ben phai (x_off>0) toe trai.
    Tra ve (R_cw, C) : R the-gioi->camera va tam camera (world).
    """
    sign = -1.0 if x_off_mm > 0 else 1.0      # xoay vao trong
    R_cw = yaw_R(sign * toe_in_deg)
    C = np.array([x_off_mm, 0.0, 0.0], dtype=np.float64)
    return R_cw, C


def project(pts_world, R_cw, C):
    """Chieu diem 3D (world) qua camera (R_cw, C) -> pixel (M,2)."""
    import cv2
    rvec, _ = cv2.Rodrigues(R_cw)
    tvec = (-R_cw @ C.reshape(3, 1))
    img, _ = cv2.projectPoints(pts_world, rvec, tvec, K, np.zeros((4, 1)))
    return img.reshape(-1, 2)


def reliable_mask(pts_world, normals_world, R_cw, C, grazing_deg):
    """
    Landmark 'dang tin' voi 1 camera khi: (1) normal huong ve camera trong gioi han
    grazing (khong nhin xien qua -> chong self-occlusion), VA (2) nam trong khung anh.
    """
    view = C[None, :] - pts_world
    view /= (np.linalg.norm(view, axis=1, keepdims=True) + 1e-9)
    facing = np.sum(normals_world * view, axis=1) > math.cos(math.radians(grazing_deg))

    px = project(pts_world, R_cw, C)
    in_fov = ((px[:, 0] >= 0) & (px[:, 0] < IMG_W) &
              (px[:, 1] >= 0) & (px[:, 1] < IMG_H))
    return facing & in_fov, px


# ------------------------------------------------------------------ rig
def build_rig(n_cam, baseline_mm, toe_in_deg):
    """N cam trai deu tren be rong 'baseline_mm'. N=2 -> +/-B/2. N=3 -> -B/2,0,+B/2."""
    if n_cam == 2:
        offs = [-baseline_mm / 2, +baseline_mm / 2]
    elif n_cam == 3:
        offs = [-baseline_mm / 2, 0.0, +baseline_mm / 2]
    else:
        offs = list(np.linspace(-baseline_mm / 2, baseline_mm / 2, n_cam))
    return [make_camera(o, toe_in_deg) for o in offs]


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def eval_config(n_cam, baseline_mm, toe_in_deg, Z_mm, noise_px, trials, grazing_deg,
                nose_depth_mm, rng, min_pair_vis=0.85):
    """
    Danh gia 1 cau hinh rig o khoang cach Z. Chay 'trials' lan (moi lan 1 bo nhieu).
    Tra ve dict: separation (median), residual that/phang, CV, visibility.
    """
    import cv2  # noqa: F401  (dam bao cv2 co truoc khi project)

    face_pts, face_nrm = face_surface(nose_depth_mm)
    flat_pts, flat_nrm = flat_surface()
    face_world = face_pts + np.array([0, 0, Z_mm])
    flat_world = flat_pts + np.array([0, 0, Z_mm])

    cams = build_rig(n_cam, baseline_mm, toe_in_deg)

    # Visibility (tren MAT THAT - phang khong bao gio self-occlude): ti le landmark
    # dang tin o TUNG cam, lay min qua cac cam (cam te nhat quyet dinh).
    masks = []
    for (R_cw, C) in cams:
        m, _ = reliable_mask(face_world, face_nrm, R_cw, C, grazing_deg)
        masks.append(m)
    per_cam_vis = [float(m.mean()) for m in masks]
    min_vis = min(per_cam_vis)

    pairs = _pairs(n_cam)
    real_res, flat_res = [], []

    for _ in range(trials):
        # Chieu 1 lan, dung chung cho moi cap; them nhieu doc lap tung view.
        proj_face = [project(face_world, R, C) + rng.normal(0, noise_px, (len(face_world), 2))
                     for (R, C) in cams]
        proj_flat = [project(flat_world, R, C) + rng.normal(0, noise_px, (len(flat_world), 2))
                     for (R, C) in cams]

        # Voi moi cap kha dung (visibility 2 dau >= nguong), tinh residual.
        pair_real, pair_flat = [], []
        for (i, j) in pairs:
            both = masks[i] & masks[j]
            if both.mean() < min_pair_vis:
                continue
            scale = float(np.ptp(proj_face[j][:, 0]))    # bat bien scale (giong 04)
            rr = _liveness.planarity_residual(proj_face[i][both], proj_face[j][both], scale=scale)
            rf = _liveness.planarity_residual(proj_flat[i][both], proj_flat[j][both], scale=scale)
            if rr and rf:
                pair_real.append(rr["mean_norm"])
                pair_flat.append(rf["mean_norm"])

        if not pair_real:
            continue
        # N>2: lay MEDIAN qua cac cap kha dung (giam variance, chong occlusion 1 cap).
        real_res.append(float(np.median(pair_real)))
        flat_res.append(float(np.median(pair_flat)))

    if not real_res:
        return dict(n_cam=n_cam, baseline=baseline_mm, toe_in=toe_in_deg, Z=Z_mm,
                    usable=False, min_vis=min_vis, per_cam_vis=per_cam_vis,
                    sep=0.0, real=0.0, flat=0.0, cv=math.inf)

    real = np.array(real_res)
    flat = np.array(flat_res)
    sep = float(np.median(real) / max(np.median(flat), 1e-9))
    cv = float(np.std(real) / max(np.mean(real), 1e-9))
    return dict(n_cam=n_cam, baseline=baseline_mm, toe_in=toe_in_deg, Z=Z_mm,
                usable=True, min_vis=min_vis, per_cam_vis=per_cam_vis,
                sep=sep, real=float(np.median(real)), flat=float(np.median(flat)), cv=cv)


# ------------------------------------------------------------------ plot
def make_plots(results, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    Zs = sorted({r["Z"] for r in results})
    angles = sorted({r["toe_in"] for r in results})
    Z_mid = Zs[len(Zs) // 2]

    # (1) Separation vs baseline (moi duong 1 goc), tai Z giua, N=2 -> tim "knee".
    plt.figure(figsize=(7, 5))
    for a in angles:
        rows = sorted([r for r in results
                       if r["n_cam"] == 2 and r["Z"] == Z_mid and r["toe_in"] == a
                       and r["usable"]], key=lambda r: r["baseline"])
        if rows:
            plt.plot([r["baseline"] for r in rows], [r["sep"] for r in rows],
                     marker="o", label=f"toe-in {a:.0f}°")
    plt.xlabel("Baseline B (mm)"); plt.ylabel("Separation (that/phang)")
    plt.title(f"Separation vs baseline (N=2, Z={Z_mid:.0f}mm)")
    plt.legend(); plt.grid(True, alpha=.3)
    p = out_dir / "sweep_sep_vs_baseline.png"; plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # (2) Visibility vs baseline (moi duong 1 goc) - tran self-occlusion.
    plt.figure(figsize=(7, 5))
    for a in angles:
        rows = sorted([r for r in results
                       if r["n_cam"] == 2 and r["Z"] == Z_mid and r["toe_in"] == a],
                      key=lambda r: r["baseline"])
        if rows:
            plt.plot([r["baseline"] for r in rows], [r["min_vis"] for r in rows],
                     marker="s", label=f"toe-in {a:.0f}°")
    plt.axhline(args.vis_thresh, color="r", ls="--", label=f"nguong {args.vis_thresh}")
    plt.xlabel("Baseline B (mm)"); plt.ylabel("Visibility (min qua 2 cam)")
    plt.title(f"Visibility vs baseline (N=2, Z={Z_mid:.0f}mm)")
    plt.legend(); plt.grid(True, alpha=.3)
    p = out_dir / "sweep_visibility.png"; plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # (3) Separation vs Z (kiem 1/Z^2), N=2, goc 0.
    plt.figure(figsize=(7, 5))
    for b in sorted({r["baseline"] for r in results}):
        rows = sorted([r for r in results
                       if r["n_cam"] == 2 and r["toe_in"] == angles[0] and r["baseline"] == b
                       and r["usable"]], key=lambda r: r["Z"])
        if len(rows) >= 2:
            plt.plot([r["Z"] for r in rows], [r["sep"] for r in rows],
                     marker="o", label=f"B={b:.0f}mm")
    plt.xlabel("Khoang cach Z (mm)"); plt.ylabel("Separation")
    plt.title(f"Separation vs khoang cach (N=2, toe-in {angles[0]:.0f}°)")
    plt.legend(); plt.grid(True, alpha=.3)
    p = out_dir / "sweep_sep_vs_distance.png"; plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    # (4) N=2 vs N=3: CV (do on dinh residual) vs baseline, tai Z giua goc 0.
    plt.figure(figsize=(7, 5))
    for n in sorted({r["n_cam"] for r in results}):
        rows = sorted([r for r in results
                       if r["n_cam"] == n and r["Z"] == Z_mid and r["toe_in"] == angles[0]
                       and r["usable"]], key=lambda r: r["baseline"])
        if rows:
            plt.plot([r["baseline"] for r in rows], [r["cv"] for r in rows],
                     marker="^", label=f"N={n}")
    plt.xlabel("Baseline B (mm)"); plt.ylabel("CV residual (thap = on dinh)")
    plt.title(f"On dinh N=2 vs N=3 (Z={Z_mid:.0f}mm, toe-in {angles[0]:.0f}°)")
    plt.legend(); plt.grid(True, alpha=.3)
    p = out_dir / "sweep_n2_vs_n3.png"; plt.savefig(p, dpi=120); plt.close(); saved.append(p)

    return saved


# ------------------------------------------------------------------ recommend
def recommend(results, vis_thresh):
    """
    Quy tac: chon cau hinh co visibility >= nguong o MOI Z (robust), roi trong so do
    lay separation (min qua Z - truong hop te nhat) cao nhat, uu tien CV thap.
    """
    Zs = sorted({r["Z"] for r in results})
    keys = sorted({(r["n_cam"], r["baseline"], r["toe_in"]) for r in results})

    best = None
    for (n, b, a) in keys:
        rows = [r for r in results if r["n_cam"] == n and r["baseline"] == b and r["toe_in"] == a]
        if len(rows) != len(Zs):
            continue
        if any((not r["usable"]) or r["min_vis"] < vis_thresh for r in rows):
            continue
        worst_sep = min(r["sep"] for r in rows)          # truong hop te nhat qua Z
        mean_cv = float(np.mean([r["cv"] for r in rows]))
        cand = dict(n=n, b=b, a=a, worst_sep=worst_sep, mean_cv=mean_cv)
        if best is None or (worst_sep, -mean_cv) > (best["worst_sep"], -best["mean_cv"]):
            best = cand
    return best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Sweep hinh hoc chon rig (N, baseline, goc)")
    ap.add_argument("--z", type=float, nargs="+", default=[450, 600, 750],
                    help="cac khoang cach nguoi dung (mm)")
    ap.add_argument("--baselines", type=float, nargs="+",
                    default=[40, 60, 80, 100, 120, 140, 160], help="cac baseline (mm)")
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 5, 10, 15, 20],
                    help="cac goc toe-in (do)")
    ap.add_argument("--ncams", type=int, nargs="+", default=[2, 3], help="cac so cam")
    ap.add_argument("--noise-px", type=float, default=1.0,
                    help="do lech chuan nhieu landmark (pixel) - PHA tautology")
    ap.add_argument("--trials", type=int, default=24, help="so lan thu moi cau hinh")
    ap.add_argument("--grazing-deg", type=float, default=70.0,
                    help="goc grazing toi da coi landmark con dang tin")
    ap.add_argument("--nose-depth", type=float, default=28.0,
                    help="do noi khoi mui-ma (mm) cua mat that gia lap")
    ap.add_argument("--vis-thresh", type=float, default=0.90,
                    help="nguong visibility toi thieu de cau hinh 'dung duoc'")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="figs", help="thu muc luu figure")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=== SWEEP HINH HOC CHON RIG (Muc A - luoi thu tuc) ===")
    print(f"Z={args.z}  baselines={args.baselines}  angles={args.angles}  "
          f"N={args.ncams}  noise={args.noise_px}px  trials={args.trials}\n")

    results = []
    for n in args.ncams:
        for b in args.baselines:
            for a in args.angles:
                for Z in args.z:
                    results.append(eval_config(
                        n, b, a, Z, args.noise_px, args.trials, args.grazing_deg,
                        args.nose_depth, rng, min_pair_vis=max(0.5, args.vis_thresh - 0.05)))

    # Bang tom tat tai Z giua.
    Z_mid = sorted(args.z)[len(args.z) // 2]
    print(f"--- Bang tai Z={Z_mid:.0f}mm (sep=tach biet, vis=visibility, CV=bien thien) ---")
    print(f"{'N':>2} {'B(mm)':>6} {'toe':>4} {'sep':>8} {'vis':>6} {'CV':>6}  usable")
    for r in sorted([r for r in results if r["Z"] == Z_mid],
                    key=lambda r: (r["n_cam"], r["baseline"], r["toe_in"])):
        mark = "OK" if (r["usable"] and r["min_vis"] >= args.vis_thresh) else "--"
        print(f"{r['n_cam']:>2} {r['baseline']:>6.0f} {r['toe_in']:>4.0f} "
              f"{r['sep']:>8.1f} {r['min_vis']:>6.2f} {r['cv']:>6.2f}  {mark}")

    best = recommend(results, args.vis_thresh)
    print("\n=== DE XUAT RIG (robust qua moi Z) ===")
    if best is None:
        print("  Khong cau hinh nao dat visibility >= nguong o MOI Z.")
        print(f"  Thu: giam --vis-thresh, giam baseline max, hoac tang --grazing-deg.")
    else:
        print(f"  So cam N        : {best['n']}")
        print(f"  Baseline B      : {best['b']:.0f} mm")
        print(f"  Goc toe-in      : {best['a']:.0f} do")
        print(f"  Separation (te nhat qua Z) : {best['worst_sep']:.1f}x")
        print(f"  CV trung binh   : {best['mean_cv']:.2f}")
        print(f"  => Baseline lon nhat con giu visibility >= {args.vis_thresh} o moi Z.")

    if not args.no_plot:
        saved = make_plots(results, args, args.out)
        print("\nDa luu figure:")
        for p in saved:
            print(f"  {p}")

    print("\nLUU Y: day la nghiem HINH HOC LY TUONG (Muc A). Coi la diem khoi dau "
          "thiet ke rig; hieu chinh lai bang pilot that.")


if __name__ == "__main__":
    main()
