# Development commands and operations

## Quick verification without models

`python scripts/reproduce_pilot_metrics.py` is `READ_ONLY`. It reproduces the archived host window counts and error rates from public numeric evidence using Python 3.10+ and the standard library.

## Application environment

Use Python 3.11 for the existing camera/inference stack. `requirements.txt` contains version ranges; it is not an exact historical paper environment lock. The inherited `requirements-pi.txt` pins direct target dependencies. Final review verified a fresh Linux AMD64 installation of those pins with compatible transitive constraints `jax==0.4.35`, `jaxlib==0.4.35`, and `opencv-contrib-python==4.10.0.84`. ARM64/Pi installation and the project Dockerfile build still need hardware-specific verification.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest httpx
python scripts/download_models.py
python main.py
```

These are `LOCAL_GENERATED` operations: environment creation, downloads, and app startup create files under `.venv/`, `models/`, `database/`, and `logs/`. The download script retrieves `buffalo_l` and `face_landmarker.task`. CPU ONNX inference is the target on Pi4.

Open `http://localhost:8000/dual-camera` for the experimental stereo path. The attendance dashboard is at `/`. For a Windows two-C270 rig, set the process environment `DUAL_CAMERA_BACKEND=dshow`; on Linux use the appropriate V4L2 camera indices. Local `python main.py` reads exported variables; copying `.env.example` to `.env` alone does not load them.

The page can start/stop paired capture. Relevant endpoints are `GET /api/dual-camera/status`, `GET /api/stereo-liveness/status`, and `POST /api/stereo-liveness/check`. The check may initialize cameras and create research logs, so it is runtime-mutating. Health/readiness endpoints are `GET /health` and `GET /ready`.

Stereo checks run in a worker thread and serialize access to the shared landmark models for one rig per process. Request overrides accept 1–30 samples and 0–10 seconds of warmup. Both host timestamps must advance between votes; frames older than `STEREO_LIVENESS_MAX_FRAME_AGE_SECONDS` (default 1 second), future timestamps, and excessive pair skew produce `INCONCLUSIVE`. Tune freshness for the actual capture rig. Unavailable numeric metrics appear as JSON `null`; saved reports use unique filenames with exclusive creation.

## Tests

```bash
python -m pytest tests/test_stereo_liveness.py -q
python -m pytest -m "not integration" -q
```

These are `TEST_RUNTIME_MUTATING`: imports may create local runtime directories and pytest may create cache. The first checks static alignment, planar/3D synthetic pairs, conservative votes, route parameters, nonblocking serialized checks, stale/repeated frames, and strict JSON/report preservation. The broader command explicitly excludes the integration test, which needs private external files and a real inference environment. No test result should be described as real-camera PAD accuracy.

On 2026-10-01, all 65 selected tests passed in the Python 3.11 Linux AMD64 review environment; one integration test was deselected. Four new regression checks failed on the pre-review implementation and passed after correction. `pip check` found no dependency conflicts. A hardware-free smoke run returned 200 for `/health`, `/ready`, `/dual-camera`, and stereo status, and confirmed that missing-camera checks return `INCONCLUSIVE` without changing attendance count. The test client emitted a Starlette deprecation warning. Model downloads, camera inference, and Pi measurements remain unverified.

Optional frontend syntax checks require Node.js:

```bash
node --check web/js/main.js
node --check web/js/scan.js
node --check web/js/enrollment.js
```

For camera smoke verification, confirm both streams, missing-face/inconclusive handling, displayed verdicts, endpoint status, and no unintended attendance creation by the stereo path. Target-device timing, temperature, RAM, camera skew, and sustained behavior need recorded hardware runs.

## Raspberry Pi source and runtime

On an intended test Pi with the correct `/dev/video0` and `/dev/video2` mappings:

```bash
docker compose config
docker compose up -d --build
docker compose logs -f
```

`config` validates/interpolates deployment configuration; `up` builds an artifact and mutates the selected machine's live runtime. It is not a command to execute on production without that operation's explicit scope. The compose file preserves `models/`, `database/`, and `logs/` through bind mounts. It disables automatic single-camera startup to avoid competing with the stereo demo. Its CPU and memory limits are requested settings; enforcement depends on the host kernel.

The app has no authentication, and evidence/photo routes expose local runtime data. Use a trusted local network. `/health` and `/ready` establish service/model readiness, not liveness efficacy.

For recovery, stop the test service and retain its mounted runtime directories. Before an upgrade, record the deployed Git revision, image identity, configuration and model versions, and prepare a consistent SQLite backup during a safe stopped/quiescent period. Roll back to the prior image/configuration only after checking database compatibility. No production deployment, backup overwrite, or runtime cleanup is part of this preparation.

## Change delivery

Use a task branch from `origin/main`, stage explicit task paths, run relevant checks, and submit a PR. Keep private data excluded. Never combine unrelated local changes into a broad `git add .`.

`docs/reference/source-manifest.json` records SHA-256 hashes for imported local software and original numeric evidence. Software imported unchanged is tagged `local-working-copy`; it is not automatically a paper-run version. The published evidence folders stay immutable; new runs use new directories.
