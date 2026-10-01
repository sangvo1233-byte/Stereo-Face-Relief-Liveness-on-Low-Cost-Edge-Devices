"""
15_measure_jitter.py - DO THUC do rung (jitter) cua landmark MediaPipe + BAO CAO ANH.

Muc dich: thay con so DOAN 0.7px bang con so DO THAT tu MediaPipe.

Cach do (bang chinh MediaPipe, khong doan):
  - Lay 1 anh mat that.
  - Tao N bien the mo phong SU KHAC BIET giua 2 frame camera cua CUNG mot mat dung yen:
      + nhieu cam bien (Gaussian)
      + nen JPEG (giong luong camera that, quality ~82 nhu config)
      + doi sang nhe (ISP/auto-exposure 2 cam khac nhau)
      + dich subpixel nho (rung tay/lech pixel)
  - Chay MediaPipe tren TUNG bien the, thu vi tri 478 landmark.
  - Do do LECH CHUAN vi tri tung landmark qua cac bien the => RUNG that (px).
  - Chuan hoa theo khoang cach 2 mat (IOD) de so sanh duoc voi mo phong.

GIOI HAN (ghi ro): day do phan RUNG NGAU NHIEN (repeatability) that cua MediaPipe.
Phan LECH CO DINH (bias) va separation stereo THAT van can 2 camera that. Nhung
rung ngau nhien la thanh phan chinh dat muc nhieu K=1 -> du de xac dinh method
o don frame manh hay yeu, va trung binh frame giup duoc bao nhieu.

Chay:
    py 15_measure_jitter.py
    py 15_measure_jitter.py --image logs/face_crops/HS001_test.jpg --variants 40
"""
import argparse
import importlib.util
import pathlib

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rig = _load("rig_sweep", "11_rig_sweep.py")

# Chi so landmark 2 khoe mat ngoai (MediaPipe canonical) - dung tinh IOD.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def make_landmarker(model_path):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def detect_px(landmarker, img_bgr):
    """Tra ve mang (478,2) toa do pixel, hoac None."""
    h, w = img_bgr.shape[:2]
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mpimg)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    return np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float64)


def perturb(img, rng, noise_std, jpeg_q, bright, shift_px):
    """Mo phong 1 frame camera khac cua CUNG mat dung yen (khong parallax)."""
    out = img.astype(np.float64)
    # dich subpixel
    dx, dy = rng.uniform(-shift_px, shift_px, 2)
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    out = cv2.warpAffine(out, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    # doi sang
    out = out + rng.uniform(-bright, bright)
    # nhieu cam bien
    out = out + rng.normal(0, noise_std, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # nen JPEG
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def separation_at_noise(noise_frac_iod, trials, rng, z=325.0, baseline=120.0, toe=3.0, nose=28.0):
    """
    Uoc luong separation khi rung = noise_frac_iod (ti le theo IOD).
    Chuyen noise_frac_iod -> px trong khong gian chieu cua rig de nhat quan.
    IOD ~ 63mm, luoi mat rong 100mm -> IOD_px ~ 0.63 * ptp(x_chieu).
    """
    face_pts, _ = rig.face_surface(nose)
    flat_pts, _ = rig.flat_surface()
    fw = face_pts + np.array([0, 0, z])
    pw = flat_pts + np.array([0, 0, z])
    cams = rig.build_rig(2, baseline, toe)
    # px noise tu ti le IOD
    ref = rig.project(fw, *cams[1])
    iod_px_sim = 0.63 * float(np.ptp(ref[:, 0]))
    npx = noise_frac_iod * iod_px_sim

    rr, rf = [], []
    for _ in range(trials):
        aL = rig.project(fw, *cams[0]) + rng.normal(0, npx, (len(fw), 2))
        aR = rig.project(fw, *cams[1]) + rng.normal(0, npx, (len(fw), 2))
        fL = rig.project(pw, *cams[0]) + rng.normal(0, npx, (len(pw), 2))
        fR = rig.project(pw, *cams[1]) + rng.normal(0, npx, (len(pw), 2))
        sr = float(np.ptp(aR[:, 0])); sf = float(np.ptp(fR[:, 0]))
        r1 = rig._liveness.planarity_residual(aL, aR, scale=sr)
        r0 = rig._liveness.planarity_residual(fL, fR, scale=sf)
        if r1 and r0:
            rr.append(r1["mean_norm"]); rf.append(max(r0["mean_norm"], 1e-12))
    if not rr:
        return 0.0
    return float(np.median(rr) / np.median(rf))


def main():
    ap = argparse.ArgumentParser(description="Do rung landmark MediaPipe that + bao cao anh")
    ap.add_argument("--image", default="logs/face_crops/HS001_test.jpg", help="anh mat")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--variants", type=int, default=40, help="so bien the nhieu")
    ap.add_argument("--noise-std", type=float, default=3.0, help="nhieu cam bien (0-255)")
    ap.add_argument("--jpeg-q", type=int, default=82, help="chat luong JPEG")
    ap.add_argument("--bright", type=float, default=6.0, help="bien do doi sang +/-")
    ap.add_argument("--shift-px", type=float, default=0.3, help="dich subpixel +/-")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    img = cv2.imread(args.image)
    if img is None:
        # thu anh mau Tom Hanks
        alt = ".conda-py311/Lib/site-packages/insightface/data/images/Tom_Hanks_54745.png"
        img = cv2.imread(alt)
        if img is None:
            raise SystemExit(f"Khong doc duoc anh: {args.image}")
        args.image = alt

    landmarker = make_landmarker(args.model)
    base_pts = detect_px(landmarker, img)
    if base_pts is None:
        raise SystemExit("Khong phat hien mat trong anh goc.")

    print("=== DO RUNG (JITTER) LANDMARK MEDIAPIPE - DU LIEU THAT ===")
    print(f"Anh: {args.image}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Bien the: {args.variants}  | nhieu {args.noise_std} + JPEG q{args.jpeg_q} "
          f"+ sang +/-{args.bright} + dich +/-{args.shift_px}px\n")

    # Thu thap landmark qua cac bien the
    stack = []
    for _ in range(args.variants):
        v = perturb(img, rng, args.noise_std, args.jpeg_q, args.bright, args.shift_px)
        pts = detect_px(landmarker, v)
        if pts is not None and pts.shape == base_pts.shape:
            stack.append(pts)
    stack = np.array(stack)  # (M, 478, 2)
    got = len(stack)
    if got < 5:
        raise SystemExit(f"Chi {got} bien the phat hien duoc mat - khong du de do.")

    # Rung tung landmark = do lech chuan vi tri qua cac bien the
    mean_pos = stack.mean(axis=0)                       # (478,2)
    dev = stack - mean_pos[None]                         # (M,478,2)
    jitter_per_lm = np.sqrt((dev ** 2).sum(axis=2)).std(axis=0)  # (478,) px RMS
    # dung: std cua khoang cach toi tam == do rung vi tri
    jitter_per_lm = np.sqrt(((stack - mean_pos[None]) ** 2).sum(axis=2).mean(axis=0))

    iod_px = float(np.linalg.norm(mean_pos[LEFT_EYE_OUTER] - mean_pos[RIGHT_EYE_OUTER]))
    jit_mean = float(jitter_per_lm.mean())
    jit_p50 = float(np.percentile(jitter_per_lm, 50))
    jit_p95 = float(np.percentile(jitter_per_lm, 95))
    jit_norm_mean = jit_mean / iod_px
    jit_norm_p95 = jit_p95 / iod_px

    print(f"Phat hien mat o {got}/{args.variants} bien the")
    print(f"IOD (khoang cach 2 mat)  : {iod_px:.1f} px")
    print(f"Rung landmark (px)       : TB={jit_mean:.2f}  p50={jit_p50:.2f}  p95={jit_p95:.2f}")
    print(f"Rung / IOD (chuan hoa)   : TB={jit_norm_mean*100:.1f}%  p95={jit_norm_p95*100:.1f}%")

    # Uoc luong separation voi rung DO DUOC (dung TB lam muc nhieu K=1)
    sep_1 = separation_at_noise(jit_norm_mean, trials=40, rng=rng)
    # trung binh K frame giam rung theo sqrt(K)
    ks = [1, 3, 5, 9, 15, 30]
    sep_k = [separation_at_noise(jit_norm_mean / np.sqrt(k), trials=40, rng=rng) for k in ks]

    print(f"\nSeparation (rung do duoc, 1 frame) : {sep_1:.1f}x")
    print(f"Separation (giu yen ~1s, 9 frame)  : {sep_k[3]:.1f}x")
    print(f"Separation (giu yen ~3s, 30 frame) : {sep_k[5]:.1f}x")

    if sep_k[3] >= 4:
        verdict = "CHAY DUOC: du bien phan biet that/gia khi giu yen ~1s + dung gan."
    elif sep_k[3] >= 2.5:
        verdict = "GIOI HAN: bien mong, can dieu kien tot (gan, du sang, giu yen)."
    else:
        verdict = "YEU: rung MediaPipe qua lon, method kho tin cay o cau hinh nay."
    print(f"\n=> VERDICT: {verdict}")

    # ---------------- BAO CAO ANH ----------------
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10))

    # A) mat + landmark to mau theo rung
    axA = fig.add_subplot(2, 2, 1)
    axA.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap="gray")
    sc = axA.scatter(mean_pos[:, 0], mean_pos[:, 1], c=jitter_per_lm, cmap="hot", s=8)
    fig.colorbar(sc, ax=axA, label="rung (px)")
    axA.set_title("A) MediaPipe kem on dinh o dau (sang = rung nhieu)")
    axA.axis("off")

    # B) histogram rung
    axB = fig.add_subplot(2, 2, 2)
    axB.hist(jitter_per_lm, bins=30, color="tab:blue", alpha=0.8)
    axB.axvline(jit_mean, color="r", ls="--", label=f"TB = {jit_mean:.2f}px")
    axB.axvline(jit_p95, color="orange", ls=":", label=f"p95 = {jit_p95:.2f}px")
    axB.set_xlabel("Rung tung landmark (px)"); axB.set_ylabel("So landmark")
    axB.set_title(f"B) Phan bo do rung (IOD={iod_px:.0f}px -> TB={jit_norm_mean*100:.1f}%)")
    axB.legend()

    # C) separation vs so frame giu yen
    axC = fig.add_subplot(2, 2, 3)
    axC.plot(ks, sep_k, marker="o", color="tab:green")
    axC.axhline(4.0, color="gray", ls="--", alpha=.6, label="nguong 'chay tot' ~4x")
    axC.axhline(1.0, color="k", alpha=.3, label="1x = khong phan biet")
    for k, s in zip(ks, sep_k):
        axC.annotate(f"{s:.1f}x", (k, s), textcoords="offset points", xytext=(0, 6), fontsize=8)
    axC.set_xlabel("So frame giu yen (K)"); axC.set_ylabel("Separation (that/phang)")
    axC.set_title(f"C) Rung do duoc -> separation (1 frame {sep_1:.1f}x)")
    axC.legend()

    # D) verdict + so sanh cam don
    axD = fig.add_subplot(2, 2, 4)
    axD.axis("off")
    lines = [
        "KET QUA DO THAT (MediaPipe)",
        "",
        f"  Rung landmark: TB {jit_mean:.2f}px  ({jit_norm_mean*100:.1f}% IOD)",
        f"  (truoc day chi DOAN ~0.7px)",
        "",
        f"  Separation 1 frame : {sep_1:.1f}x",
        f"  Giu yen ~1s (9fr)  : {sep_k[3]:.1f}x",
        f"  Giu yen ~3s (30fr) : {sep_k[5]:.1f}x",
        "",
        f"  => {verdict}",
        "",
        "SO VOI CAM DON:",
        "  Cam don: doan qua VE NGOAI (texture,",
        "    moire) -> anh in NET co the lot.",
        "  Cam doi: do KHOI 3D that -> anh phang",
        "    bi bat bat ke net/dep. Khong can train.",
        "",
        "CON THIEU (can 2 cam that):",
        "  - lech co dinh (bias) cua MediaPipe",
        "  - separation stereo do truc tiep",
    ]
    axD.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10,
             family="monospace", transform=axD.transAxes)

    fig.suptitle("BAO CAO DO RUNG MEDIAPIPE + DANH GIA METHOD STEREO",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / "jitter_report.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\nDa luu bao cao anh: {p}")


if __name__ == "__main__":
    main()
