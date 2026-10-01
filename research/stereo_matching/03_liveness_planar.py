"""
Buoc tinh 3: LIVENESS bang kiem tra DO PHANG tu mot cap anh stereo.

Y tuong cot loi (KHONG can calib, KHONG can triangulate o giai doan nay):
  - Mot mat phang (anh in, man hinh dien thoai) anh xa giua 2 view bang DUNG MOT
    homography H. Neu fit 1 homography qua TAT CA landmark va sai so nho
    -> be mat phang -> SPOOF.
  - Mat that co khoi 3D (mui nho ra, tai lui vao) -> khong mot homography nao
    fit het duoc -> sai so tai chieu LON -> mat that.

Day la phep thu re va manh: chi can cap landmark tuong ung, khong can ma tran camera.
Sai so duoc CHUAN HOA theo khoang cach 2 mat (inter-ocular) de khong phu thuoc
mat o gan hay xa, anh to hay nho.

Chay:
    python 03_liveness_planar.py trai.jpg phai.jpg
    python 03_liveness_planar.py trai.jpg phai.jpg --thresh 0.02
"""

import argparse
import sys

import cv2
import numpy as np

from face_utils import matched_points, interocular_distance


def planarity_residual(ptsL, ptsR, scale=None):
    """
    Fit homography ptsL->ptsR (RANSAC), tra ve sai so tai chieu trung binh
    da CHUAN HOA theo do dai tham chieu. Cang nho = cang phang.

    scale: do dai dung de chuan hoa (pixel). Mac dinh = inter-ocular distance
           (hop voi landmark MediaPipe 468 diem). Khi test tong hop khong co
           landmark mat that, truyen scale thu cong.
    """
    H, mask = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        return None

    # Tai chieu toan bo landmark trai sang phai bang H, do sai lech voi thuc te.
    src = ptsL.reshape(-1, 1, 2).astype(np.float64)
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(proj - ptsR, axis=1)  # sai so tung diem (pixel)

    iod = scale if scale is not None else interocular_distance(ptsR)
    if iod < 1e-6:
        return None

    return {
        "mean_norm": float(np.mean(err) / iod),     # chuan hoa
        "p95_norm": float(np.percentile(err, 95) / iod),
        "mean_px": float(np.mean(err)),
        "max_px": float(np.max(err)),
        "iod_px": iod,
        "inlier_ratio": float(mask.mean()) if mask is not None else 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Liveness bang do phang (homography)")
    ap.add_argument("left", help="anh cam trai")
    ap.add_argument("right", help="anh cam phai")
    ap.add_argument("--thresh", type=float, default=0.005,
                    help="nguong sai so chuan hoa: > nguong => mat that (mac dinh 0.005). "
                         "PHAI tinh chinh bang du lieu that/gia tren chinh phan cung cua ban.")
    args = ap.parse_args()

    il = cv2.imread(args.left)
    ir = cv2.imread(args.right)
    if il is None or ir is None:
        sys.exit("Khong doc duoc mot trong hai anh.")

    ptsL, ptsR = matched_points(il, ir)
    if ptsL is None:
        sys.exit("Khong phat hien du khuon mat o ca hai anh.")

    r = planarity_residual(ptsL, ptsR)
    if r is None:
        sys.exit("Khong fit duoc homography.")

    print(f"Inter-ocular (anh phai) : {r['iod_px']:.1f} px")
    print(f"Sai so tai chieu TB      : {r['mean_px']:.2f} px")
    print(f"Sai so chuan hoa (TB)    : {r['mean_norm']:.4f}")
    print(f"Sai so chuan hoa (p95)   : {r['p95_norm']:.4f}")
    print(f"Ti le inlier homography  : {r['inlier_ratio']:.2f}")

    live = r["mean_norm"] > args.thresh
    print("=> " + ("MAT THAT (co khoi 3D, khong fit 1 mat phang)" if live
                   else "NGHI SPOOF (be mat phang - anh in/man hinh)"))
    print(f"   (nguong = {args.thresh}; can tinh chinh bang du lieu that/gia thuc te)")


if __name__ == "__main__":
    main()
