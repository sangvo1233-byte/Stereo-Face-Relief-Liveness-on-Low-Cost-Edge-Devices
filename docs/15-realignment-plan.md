> **Historical record (2026-06-25).** This document preserves an earlier migration and its decisions. Its scaffold paths, cleanup proposals, machine-local paths, and pending task list are historical context. For the current research companion, start with `README.md`, `docs/architecture.md`, `docs/reproduction.md`, and `docs/development.md`.

# 15 - Ke hoach sap xep lai IOT-project theo face-reg-finnal-project

Cap nhat: 2026-06-25. Tai lieu nay ghi lai SU THAT ve du an + ke hoach don dep,
de tranh quen ve sau. DOC TRUOC KHI THUC THI.

## 0. Boi canh va su that quan trong

- Project THAT cua minh la `C:\Users\ADMIN\Desktop\Projects\face-reg-finnal-project`.
  Day moi la he thong se nhung vao Raspberry Pi. Day la MUC DICH that cua du an.
- `IOT-project` (repo hien tai) la ban scaffold cu, KHONG bam sat project that:
  - Tu bia ra model YuNet + SFace (ONNX nhe) - KHONG co trong project that.
  - Tu bia ra backend "mock", factory chon backend, va nhieu doc suy doan
    (11-14) ve hieu suat/feasibility dua tren model sai.
  - Ket luan FPS, soak test, arm-compat... deu neo vao model sai => khong dung.
- => Can don dep IOT-project, dong bo lai theo project that, roi port len Pi.

## 1. Su that ve project that (face-reg-finnal-project)

Stack: FastAPI + InsightFace (ArcFace) + MediaPipe + SQLite + dashboard JS thuan.

Model THAT dang dung:
- InsightFace goi `buffalo_l` (nap qua FaceAnalysis):
  - `det_10g.onnx` (~16 MB) - phat hien khuon mat (SCRFD), tra 5 landmark.
  - `w600k_r50.onnx` (~166 MB) - ArcFace ResNet50, embedding 512-D.
  - `2d106det.onnx`, `1k3d68.onnx`, `genderage.onnx` - di kem goi, khong phai loi.
- MediaPipe `face_landmarker.task` (~3.6 MB) - landmark day cho liveness/challenge
  (chop mat EAR, tu the dau, moire).

Tham so loi (config.py): EMBEDDING_DIM=512, COSINE_THRESHOLD=0.45,
DET_SIZE=(480,480), INSIGHTFACE_MODEL="buffalo_l".

Tinh nang chinh:
- Detect V4.4 (mac dinh): scan qua browser stream (ws/scan-v4) + local webcam
  (ws/scan-v4/local), runtime core/runtime_v4.py + local_runner_v4.py.
- Detect V3 (compat): strict + moire + active challenge.
- Legacy V1: 1 frame /api/scan, /api/enroll.
- Enroll V2 da goc (front/left/right), quality gate.
- Anti-spoofing nhieu lop: moire, screen-context, phone-rectangle, passive
  liveness, stream liveness, active challenge (TURN_LEFT/RIGHT, LOOK_UP/DOWN,
  OPEN_MOUTH, CENTER_HOLD).
- DB SQLite (student/session/attendance), evidence anh, dashboard web/.

Dependency that (requirements.txt): opencv-python, numpy, onnxruntime,
insightface, mediapipe, fastapi, uvicorn[standard], python-multipart, loguru,
scikit-learn, filterpy, scipy, jinja2.

## 2. Pha A - Don dep nhung gi minh lam sai trong IOT-project

Muc tieu: xoa het thu tu bia, giu lai bo khung repo de tai cau truc.

Can XOA (vi dua tren model/ kien truc sai):
- device/src/pipeline/onnx_backend.py, factory.py, stages.py (mock), recognizer.py
  -> se thay bang code that port tu project that o Pha C.
- device/models/*.onnx (YuNet/SFace) + scripts lien quan model sai:
  download_models.py, export_models.py, install_models.py.
- scripts benchmark/feasibility neo vao model sai: benchmark_pipeline.py,
  perf_report.py, load_estimate_pi.py, soak_test.py, eval_accuracy.py,
  check_arm_wheels.py, compat_check.py, recognize_demo.py, run_compat_docker.ps1.
- docs suy doan sai: 10-reference-system-review, 11-testing-on-pc,
  12-real-recognition-on-pc, 13-arm-compat-status, 14-pi-feasibility-verdict,
  + perf_pc.json. (Giu lai docs goc 00-09 nhung se sua o Pha B.)
- dist/ (bundle + model offline cu), docs/samples (anh mau khong lien quan).
- server/ scaffold cu (FastAPI + SQLite don gian) neu trung lap voi project that.

Can GIU:
- .git (lich su), .gitignore, .gitattributes (cap nhat lai), AGENTS.md, README.md
  (viet lai), .kiro/ (xem co con dung khong).

Luu y git: cac commit minh da tao (f9f009b...) chua push len remote. Se KHONG
push trang thai sai nay. Quyet dinh o muc 6 (lich su git).

## 3. Pha B - Dong bo kien truc & tai lieu

Viet lai docs cho dung project that:
- 01-overview: mo ta he FastAPI + InsightFace + MediaPipe + anti-spoofing.
- 02-requirements: chuc nang that (enroll da goc, scan v4.4, liveness, challenge,
  attendance, dashboard).
- 03-architecture: so do camera/browser -> detect+embed -> liveness layers ->
  match -> DB -> dashboard. Ghi ro client (browser/phone) + server.
- 05-pipeline-design: pipeline Detect V4.4 that (8 buoc nhu README project that).
- 06-server-api: liet ke routes that (attendance, enrollment, enrollment_v2,
  scan_v3, scan_v4, local_scan, live, phone_camera, system).
- 04-hardware-bom: cap nhat cho Pi 4 + camera, lieu ArcFace 166MB co kha thi.
- Xoa/thay cac doc 10-14 cu bang doc moi dung thuc te.
- AGENTS.md + README.md: cap nhat mo ta dung, tro toi project that la goc.

## 4. Pha C - Port that sang Pi

Day la pha lon, can danh gia kha thi truoc khi code.

Buoc C1 - Danh gia kha thi tren Pi 4 (QUAN TRONG):
- ArcFace w600k_r50 (~166MB, ResNet50, embed 512-D) NANG hon SFace nhieu.
  Tren Pi 4 CPU co the cham (vai tram ms/mat). Can do that hoac uoc luong tu
  benchmark cong khai. Neu qua cham -> can phuong an:
  - dung model InsightFace nho hon (vd buffalo_s: det_500m + w600k_mbf), hoac
  - quantize INT8, hoac giam DET_SIZE, frame skipping.
- MediaPipe face_landmarker chay tren Pi: kiem tra wheel ARM (mediapipe co
  aarch64 wheel cho Python nao). Day la rui ro tuong thich lon.
- onnxruntime + insightface tren aarch64: kiem tra wheel that su (lan truoc
  minh chi kiem cho YuNet/SFace, gio doi model + them mediapipe/scikit/filterpy).

Buoc C2 - Cau truc port:
- Quyet dinh: copy NGUYEN project that vao IOT-project (1 nguon su that) hay
  giu IOT-project nhu ban "edge-optimized" rieng. Khuyen nghi: hop nhat thanh
  MOT project that, IOT-project tro thanh repo deploy chinh thuc.
- Mang code that: config.py, main.py, core/, app/, web/, tests/, requirements.txt.
- Bo phan dev/ (lich su nghien cuu) khoi ban deploy, giu rieng neu can.

Buoc C3 - Thich nghi Pi:
- Camera: server webcam (local_runner_v4) qua OpenCV/V4L2 hoac Picamera2.
- Script setup_pi.sh moi: cai deps that, tai buffalo_l + face_landmarker.task.
- systemd service chay uvicorn main:app.
- Cau hinh CAMERA_*, providers chi CPU (bo CUDA tren Pi).

Buoc C4 - Verify tren Pi: do FPS that, nhiet, liveness hoat dong, end-to-end
enroll -> scan -> attendance -> dashboard.

## 5. Pha D - Sap xep thanh project hoan chinh

- Cau truc thu muc cuoi (de xuat):
  - config.py, main.py, requirements.txt, README.md, .env.example
  - app/ (routes), core/ (engine, camera, liveness, runtime), web/ (dashboard)
  - database/ (schema, KHONG commit .db), models/ (gitignore, tai qua script)
  - scripts/ (setup_pi, download_models, run service), docs/, tests/
- .gitignore chuan: models/, *.db, logs/, evidence, face_crops, .env, .venv.
- Loai bo file rac: logs cu, backups db, anh nguoi that, dev iterations nang.
- README mot nguon: cai dat PC + deploy Pi + kien truc + API.

## 6. Quyet dinh can user chot truoc khi thuc thi

1. Hop nhat hay tach: dua code that vao IOT-project (1 repo), hay tao repo moi?
   (Anh huong remote git da cau hinh: sangvo1233-byte/IOT-project.)
2. Lich su git: giu cac commit scaffold sai (ghi de bang commit don dep) hay
   lam lai lich su sach (vd git checkout --orphan)? Lua chon sau la pha huy,
   can user dong y.
3. Pha C kha thi: neu ArcFace 166MB qua nang/MediaPipe khong co wheel ARM,
   user chap nhan doi sang model nhe hon (buffalo_s/INT8) khong?
4. Du lieu nhay cam: anh nguoi that + DB trong project that KHONG duoc push
   len GitHub public. Xac nhan chi mang code + script tai model.

## 7. Thu tu thuc thi de xuat

A (don dep) -> B (dong bo doc) -> C1 (danh gia kha thi Pi) -> [chot lai voi user]
-> C2/C3 (port + thich nghi) -> C4 (verify) -> D (sap xep + .gitignore) -> push.

Khong push len remote cho den khi xong Pha D va user duyet.

## 8. QUYET DINH DA CHOT (2026-06-25)

1. Hop nhat vao IOT-project (dung lai remote sangvo1233-byte/IOT-project). KHONG tao repo moi.
2. Lam lai lich su git SACH: 1 commit khoi tao moi (chua push). Bo 3 commit scaffold cu.
3. Giu NGUYEN 2 model that (InsightFace buffalo_l + MediaPipe face_landmarker.task)
   de nhung thu len Pi va do hieu suat that. CHUA doi sang model nhe.
4. KHONG push du lieu nhay cam: anh nguoi that (logs/face_crops, logs/evidence),
   *.db, database/backups, .env, logs, models nang. Chi mang code + script tai model.

Pha A + merge thuc hien ngay. Pha C (port/adapt Pi) va Pha B (viet lai doc chi tiet)
lam tiep sau khi co cau truc sach + test pass.
