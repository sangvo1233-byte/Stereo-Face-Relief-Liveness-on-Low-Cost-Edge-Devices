"""
17_epipolar_liveness.py - LIVENESS bang so sanh 2 MO HINH HINH HOC (khong do residual don le).

DOT PHA so voi 03-16:
  Suot buoi ta do "residual sau homography" va thay no la tin hieu tồi (nhieu loan,
  artifact). Sua dung KHONG phai do residual homography — ma HOI: MO HINH NAO fit?

  - Homography H: chi fit khi canh PHANG hoac camera chi XOAY (khong dich).
  - Fundamental F: fit khi canh 3D tong quat + camera DICH.

  Chu ky phan biet:
    * Anh PHANG (spoof)       : H fit tot            -> SPOOF
    * Mat 3D that + cam dich  : H fail, F fit den san nhieu -> LIVE
    * Non-rigid / sai diem    : ca hai fail          -> KHONG KET LUAN

QUAN TRONG (dong lo F-suy-bien):
  Trong hinh hoc da goc nhin, F KHONG xac dinh duy nhat khi canh phang. Nghia la
  anh phang co the lam F "cung co ve fit". Nen ta kiem H TRUOC: neu H fit -> phang
  -> SPOOF, BAT KE F noi gi. Chi khi H fail moi xet F. Thu tu nay chong duoc spoof.

  F fit "den san nhieu" moi la bang chung 3D: khong the fit thap hon nhieu landmark.
  Neu Sampson >> san nhieu thi la sai correspondence, khong phai 3D sach.

Chay:
    py 17_epipolar_liveness.py --left C:/Users/ADMIN/Desktop/1.jpg --right C:/Users/ADMIN/Desktop/2.jpg
    py 17_epipolar_liveness.py --self-test   # test nhanh SPOOF bang warp phang tong hop
"""
import argparse
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Nguong (chuan hoa theo IOD, don vi %)
INLIER_FIT = 0.70        # >= coi la mo hinh "fit"
INLIER_FAIL = 0.40       # <  coi la mo hinh "fail"
NEAR_NOISE = 2.0         # residual/Sampson <= 2% IOD coi la "den san nhieu"


def make_landmarker(model_path):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def detect_px(landmarker, img_bgr):
    h, w = img_bgr.shape[:2]
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mpimg)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]
    return np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float64)


def sampson_distance(F, p1, p2):
    """Khoang cach Sampson (xap xi bac nhat cua sai so tai chieu epipolar), don vi px."""
    p1h = np.hstack([p1, np.ones((len(p1), 1))])
    p2h = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = (F @ p1h.T).T
    Ftx2 = (F.T @ p2h.T).T
    num = np.sum(p2h * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def classify(ptsL, ptsR):
    """Tra ve dict day du: fit ca H va F, phan loai theo thu tu H-truoc."""
    iod = np.linalg.norm(ptsR[RIGHT_EYE_OUTER] - ptsR[LEFT_EYE_OUTER])
    thr = 0.02 * iod  # nguong RANSAC = 2% IOD

    # --- Mo hinh H ---
    H, maskH = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, thr)
    projH = cv2.perspectiveTransform(ptsL.reshape(-1, 1, 2), H).reshape(-1, 2)
    resH = np.linalg.norm(projH - ptsR, axis=1)
    inlierH = float(maskH.mean())
    resH_pct = resH.mean() / iod * 100

    # --- Mo hinh F ---
    F, maskF = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, thr, 0.999)
    if F is None or F.shape != (3, 3):
        inlierF, sampson_pct, sampson_p95 = 0.0, float("inf"), float("inf")
    else:
        sd = sampson_distance(F, ptsL, ptsR)
        inlierF = float(maskF.mean())
        sampson_pct = sd.mean() / iod * 100
        sampson_p95 = np.percentile(sd, 95) / iod * 100

    # --- Phan loai (H TRUOC de dong lo F-suy-bien tren canh phang) ---
    if inlierH >= INLIER_FIT and resH_pct <= NEAR_NOISE:
        verdict = "SPOOF"
        reason = (f"H fit tot (inlier {inlierH:.2f}, residual {resH_pct:.2f}% IOD den san nhieu) "
                  f"-> canh PHANG. Anh in/man hinh. Bo qua F du F co fit (F suy bien tren mat phang).")
    elif inlierH < INLIER_FAIL and inlierF >= INLIER_FIT and sampson_pct <= NEAR_NOISE:
        verdict = "LIVE"
        reason = (f"H FAIL (inlier {inlierH:.2f}) nhung F FIT den san nhieu "
                  f"(inlier {inlierF:.2f}, Sampson {sampson_pct:.2f}% IOD) "
                  f"-> co cau truc 3D that + camera dich. Mat that.")
    elif inlierF >= INLIER_FIT and sampson_pct <= NEAR_NOISE and inlierH < INLIER_FIT:
        verdict = "LIVE (yeu)"
        reason = (f"F fit den san nhieu (inlier {inlierF:.2f}, Sampson {sampson_pct:.2f}%) "
                  f"trong khi H khong fit han (inlier {inlierH:.2f}). Nghieng ve 3D that "
                  f"nhung lech goc chua du ro; nen xac nhan them.")
    else:
        verdict = "KHONG KET LUAN"
        reason = (f"Ca H (inlier {inlierH:.2f}, res {resH_pct:.1f}%) lan F "
                  f"(inlier {inlierF:.2f}, Sampson {sampson_pct:.1f}%) deu khong fit sach "
                  f"-> non-rigid (doi bieu cam) hoac sai correspondence. Can cap tot hon.")

    return {
        "iod": iod, "inlierH": inlierH, "resH_pct": resH_pct,
        "inlierF": inlierF, "sampson_pct": sampson_pct, "sampson_p95": sampson_p95,
        "verdict": verdict, "reason": reason,
    }


def print_report(r, title):
    print(f"\n=== {title} ===")
    print(f"IOD (phai)            : {r['iod']:.1f} px")
    print(f"Mo hinh H  inlier@2%  : {r['inlierH']:.2f}   residual TB: {r['resH_pct']:.2f}% IOD")
    print(f"Mo hinh F  inlier@2%  : {r['inlierF']:.2f}   Sampson  TB: {r['sampson_pct']:.2f}% IOD "
          f"(p95 {r['sampson_p95']:.2f}%)")
    print(f"\n=> VERDICT: {r['verdict']}")
    print(f"   {r['reason']}")


def self_test_spoof(lm, img_path):
    """Test nhanh nhanh SPOOF: warp anh qua 1 homography PHANG -> ep 'goc thu 2 cua mat phang'.
    Ky vong: H fit tot -> phan loai SPOOF. Dong vong discriminator."""
    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Khong doc duoc {img_path}")
    h, w = img.shape[:2]
    # Homography phang mo phong camera lech goc: keo 4 goc anh
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.05*w, 0.02*h], [0.95*w, 0.06*h],
                      [0.90*w, 0.98*h], [0.08*w, 0.94*h]])
    Hplanar = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Hplanar, (w, h))

    ptsL = detect_px(lm, img)
    ptsR = detect_px(lm, warped)
    if ptsL is None or ptsR is None:
        raise SystemExit(f"Warp phang: khong detect duoc mat "
                         f"(L={'OK' if ptsL is not None else 'FAIL'} "
                         f"R={'OK' if ptsR is not None else 'FAIL'})")
    r = classify(ptsL, ptsR)
    print_report(r, "SELF-TEST SPOOF (warp phang tong hop)")
    ok = r["verdict"] == "SPOOF"
    print(f"\n   KY VONG SPOOF -> {'PASS' if ok else 'FAIL: ' + r['verdict']}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Liveness bang so sanh mo hinh H vs F")
    ap.add_argument("--left", default="C:/Users/ADMIN/Desktop/1.jpg")
    ap.add_argument("--right", default="C:/Users/ADMIN/Desktop/2.jpg")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--self-test", action="store_true",
                    help="Chay test nhanh SPOOF bang warp phang tong hop")
    args = ap.parse_args()

    lm = make_landmarker(args.model)

    if args.self_test:
        self_test_spoof(lm, args.left)
        return

    imgL = cv2.imread(args.left)
    imgR = cv2.imread(args.right)
    if imgL is None or imgR is None:
        raise SystemExit("Khong doc duoc mot trong hai anh.")
    ptsL = detect_px(lm, imgL)
    ptsR = detect_px(lm, imgR)
    if ptsL is None or ptsR is None:
        raise SystemExit(f"Khong detect duoc mat: L={'OK' if ptsL is not None else 'FAIL'} "
                         f"R={'OK' if ptsR is not None else 'FAIL'}")
    r = classify(ptsL, ptsR)
    print_report(r, "LIVENESS TREN CAP ANH THAT")


if __name__ == "__main__":
    main()
