"""
Module dung chung: trich landmark khuon mat tu anh tinh bang MediaPipe Face Mesh.

Vi sao quan trong cho stereo:
  MediaPipe tra ve 468 landmark da duoc DANH SO theo ngu nghia. Landmark so i
  ben anh trai = landmark so i ben anh phai (cung diem giai phau). Nho vay bai
  toan "tim diem tuong ung giua 2 anh" (correspondence) - cho kho nhat cua stereo
  - duoc giai MIEN PHI, khong can block matching tren da tron.

Cai dat:
    pip install mediapipe opencv-python numpy
"""

import sys

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("Thieu mediapipe. Cai: pip install mediapipe opencv-python numpy")


# Subset 6 diem on dinh cho solvePnP (head pose). Khop voi MODEL_POINTS_3D o 01.
POSE_LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

# Cap mat de chuan hoa khoang cach (inter-ocular). Dung lam don vi do "kich thuoc mat".
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def detect_landmarks(image_bgr, max_faces=1):
    """
    Tra ve mang (468, 2) toa do pixel cua landmark khuon mat dau tien,
    hoac None neu khong thay mat.
    """
    h, w = image_bgr.shape[:2]
    mp_mesh = mp.solutions.face_mesh
    with mp_mesh.FaceMesh(static_image_mode=True,
                          max_num_faces=max_faces,
                          refine_landmarks=True,
                          min_detection_confidence=0.5) as mesh:
        res = mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not res.multi_face_landmarks:
        return None

    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float64)
    return pts


def interocular_distance(pts):
    """Khoang cach 2 khoe mat ngoai (pixel). Dung de chuan hoa sai so theo kich thuoc mat."""
    return float(np.linalg.norm(pts[LEFT_EYE_OUTER] - pts[RIGHT_EYE_OUTER]))


def approx_intrinsics(image_shape):
    """
    Ma tran noi (K) uoc luong khi chua calib: tieu cu ~ chieu rong anh,
    tam quang tai giua anh, bo qua meo. Du dung cho giai doan nghien cuu tinh.
    """
    h, w = image_shape[:2]
    focal = float(w)
    K = np.array([[focal, 0, w / 2.0],
                  [0, focal, h / 2.0],
                  [0, 0, 1]], dtype=np.float64)
    return K


def matched_points(img_left, img_right):
    """
    Tra ve (ptsL, ptsR) la 2 mang (468,2) landmark tuong ung giua 2 anh,
    hoac (None, None) neu mot trong hai anh khong co mat.
    Vi cung 1 mat -> ca 2 deu co du 468 diem cung chi so -> tuong ung 1-1.
    """
    pl = detect_landmarks(img_left)
    pr = detect_landmarks(img_right)
    if pl is None or pr is None:
        return None, None
    return pl, pr
