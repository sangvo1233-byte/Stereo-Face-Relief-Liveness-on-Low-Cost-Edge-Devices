import math
from pathlib import Path
import sys

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stereo_liveness import (  # noqa: E402
    VERDICT_INCONCLUSIVE,
    VERDICT_LIVE,
    VERDICT_SPOOF,
    aggregate_verdict,
    apply_spoof_suppression,
    classify_landmark_pair,
)
from core.stereo_alignment import align_stereo_pair, alignment_metadata  # noqa: E402


def _face_like_points(count: int = 468) -> np.ndarray:
    rng = np.random.default_rng(42)
    pts = rng.normal(loc=(320.0, 240.0), scale=(75.0, 95.0), size=(count, 2))
    pts[:, 0] = np.clip(pts[:, 0], 120.0, 520.0)
    pts[:, 1] = np.clip(pts[:, 1], 70.0, 410.0)
    pts[33] = (260.0, 210.0)
    pts[263] = (380.0, 210.0)
    return pts.astype(np.float64)


def test_static_alignment_outputs_clean_640x452_pair():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    left[120, 320] = (255, 255, 255)
    right[148, 320] = (255, 255, 255)

    aligned_left, aligned_right = align_stereo_pair(left, right)
    meta = alignment_metadata()

    assert aligned_left.shape == (452, 640, 3)
    assert aligned_right.shape == (452, 640, 3)
    assert meta["output_width"] == 640
    assert meta["output_height"] == 452
    assert aligned_left[120, 320].tolist() == [255, 255, 255]
    assert aligned_right[120, 320].tolist() == [255, 255, 255]


def test_planar_homography_pair_is_spoof():
    pts_left = _face_like_points()
    h_mat = cv2.getPerspectiveTransform(
        np.float32([[120, 70], [520, 80], [500, 410], [140, 400]]),
        np.float32([[135, 78], [508, 95], [486, 402], [154, 388]]),
    )
    pts_right = cv2.perspectiveTransform(pts_left.reshape(-1, 1, 2), h_mat).reshape(-1, 2)

    result = classify_landmark_pair(pts_left, pts_right)

    assert result["verdict"] == VERDICT_SPOOF
    assert result["metrics"]["h_inlier"] >= 0.95
    assert result["metrics"]["h_residual_pct"] < 0.2


def test_clean_3d_pair_can_vote_live():
    rng = np.random.default_rng(7)
    n = 468
    x = rng.normal(0.0, 42.0, n)
    y = rng.normal(0.0, 58.0, n)
    relief = 80.0 * np.exp(-((x / 38.0) ** 2 + ((y + 8.0) / 46.0) ** 2))
    z = 620.0 - relief + rng.normal(0.0, 1.0, n)
    pts3d = np.column_stack([x, y, z])
    pts3d[33] = (-58.0, -18.0, 620.0)
    pts3d[263] = (58.0, -18.0, 620.0)

    f_px = 720.0
    cx, cy = 320.0, 240.0
    baseline = 100.0

    def project(camera_x: float) -> np.ndarray:
        xc = pts3d[:, 0] - camera_x
        yc = pts3d[:, 1]
        zc = pts3d[:, 2]
        return np.column_stack([f_px * xc / zc + cx, f_px * yc / zc + cy])

    pts_left = project(-baseline / 2.0)
    pts_right = project(baseline / 2.0)

    result = classify_landmark_pair(pts_left, pts_right)

    assert result["verdict"] == VERDICT_LIVE
    assert result["metrics"]["f_inlier"] >= 0.60
    assert math.isfinite(result["metrics"]["f_sampson_pct"])


def test_temporal_vote_fails_safe_on_conflict():
    samples = [
        {"verdict": VERDICT_LIVE},
        {"verdict": VERDICT_LIVE},
        {"verdict": VERDICT_SPOOF},
        {"verdict": VERDICT_INCONCLUSIVE},
        {"verdict": VERDICT_INCONCLUSIVE},
    ]

    result = aggregate_verdict(samples, min_votes=2)

    assert result["verdict"] == VERDICT_INCONCLUSIVE
    assert result["votes"]["live"] == 2
    assert result["votes"]["spoof"] == 1


def test_single_live_sample_is_not_enough_for_summary_live():
    result = aggregate_verdict([{"verdict": VERDICT_LIVE}], min_votes=1, min_live_votes=3)

    assert result["verdict"] == VERDICT_INCONCLUSIVE
    assert result["votes"]["live"] == 1
    assert result["votes"]["min_live_votes"] == 3


def test_single_spoof_sample_can_block_fast():
    result = aggregate_verdict([{"verdict": VERDICT_SPOOF}], min_votes=1, min_live_votes=3)

    assert result["verdict"] == VERDICT_SPOOF
    assert result["votes"]["spoof"] == 1


def test_recent_spoof_suppresses_following_live(monkeypatch):
    import core.stereo_liveness as stereo_liveness

    monkeypatch.setattr(stereo_liveness.config, "STEREO_LIVENESS_SPOOF_SUPPRESS_SECONDS", 3.0)
    monkeypatch.setattr(stereo_liveness, "_SPOOF_SUPPRESS_UNTIL", 0.0)

    spoof_summary = aggregate_verdict([{"verdict": VERDICT_SPOOF}], min_votes=1, min_live_votes=3)
    apply_spoof_suppression(spoof_summary, [{"verdict": VERDICT_SPOOF}])
    live_summary = aggregate_verdict(
        [{"verdict": VERDICT_LIVE}, {"verdict": VERDICT_LIVE}, {"verdict": VERDICT_LIVE}],
        min_votes=3,
        min_live_votes=3,
    )
    result = apply_spoof_suppression(live_summary, [{"verdict": VERDICT_LIVE}])

    assert result["verdict"] == VERDICT_INCONCLUSIVE
    assert result["reason"] == "recent planar spoof evidence suppresses LIVE"




def test_stereo_liveness_route_passes_parameters(monkeypatch):
    from app.routes import dual_camera

    seen = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return {"success": True, "summary": {"verdict": VERDICT_INCONCLUSIVE}}

    monkeypatch.setattr(dual_camera, "run_stereo_liveness_check", fake_check)
    app = FastAPI()
    app.include_router(dual_camera.router)

    client = TestClient(app)
    res = client.post("/api/stereo-liveness/check?sample_count=4&save_report=false&warmup_seconds=0.5")

    assert res.status_code == 200
    assert res.json()["summary"]["verdict"] == VERDICT_INCONCLUSIVE
    assert seen == {"sample_count": 4, "save_report": False, "warmup_seconds": 0.5}


def test_stereo_check_does_not_block_status_or_overlap_shared_models(monkeypatch):
    import asyncio
    import threading
    import httpx
    from app.routes import dual_camera

    entered, release = threading.Event(), threading.Event()
    active = []

    def slow_check(**kwargs):
        active.append(1)
        assert len(active) == 1
        entered.set()
        release.wait(timeout=1)
        active.pop()
        return {"success": True}

    monkeypatch.setattr(dual_camera, "run_stereo_liveness_check", slow_check)
    app = FastAPI()
    app.include_router(dual_camera.router)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/api/stereo-liveness/check"))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert active, "Blocking check finished before the event loop could respond"
                status = await asyncio.wait_for(client.get("/api/stereo-liveness/status"), 0.5)
                assert status.status_code == 200
                second = asyncio.create_task(client.post("/api/stereo-liveness/check"))
                await asyncio.sleep(0.05)
                release.set()
                assert (await first).status_code == 200
                assert (await second).status_code == 200
            finally:
                release.set()
                await first

    asyncio.run(exercise())


def test_stereo_check_rejects_stale_and_repeated_frames(monkeypatch):
    import time
    from types import SimpleNamespace
    import core.stereo_liveness as stereo

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(stereo, "analyze_frame_pair", lambda *a: {"verdict": VERDICT_LIVE})
    for timestamp in (time.time() - 10, time.time(), time.time() + 10, float("nan")):
        camera = SimpleNamespace(get_latest_frame_with_timestamp=lambda copy, ts=timestamp: (frame, ts))
        dual = SimpleNamespace(start=lambda: None, get_camera=lambda side: camera, get_status=lambda: {})
        result = stereo.run_stereo_liveness_check(
            dual=dual, sample_count=3, min_votes=3, sample_interval_seconds=0, save_report=False, warmup_seconds=0,
        )
        assert result["summary"]["verdict"] == VERDICT_INCONCLUSIVE
    camera = SimpleNamespace(get_latest_frame_with_timestamp=lambda copy: (frame, time.time()))
    result = stereo.run_stereo_liveness_check(
        dual=dual, sample_count=3, min_votes=3, sample_interval_seconds=0, save_report=False, warmup_seconds=0,
    )
    assert result["summary"]["verdict"] == VERDICT_LIVE


def test_unavailable_epipolar_fit_has_json_safe_metrics(monkeypatch, tmp_path):
    import json
    import core.stereo_liveness as stereo

    monkeypatch.setattr(cv2, "findFundamentalMat", lambda *args: (None, None))
    points = _face_like_points()
    result = classify_landmark_pair(points, points)
    assert result["verdict"] == VERDICT_SPOOF
    json.dumps(result, allow_nan=False)
    monkeypatch.setattr(stereo.config, "LOGS_DIR", tmp_path)
    first, second = stereo._save_report(result), stereo._save_report(result)
    assert first != second
    assert json.loads(first.read_text())["metrics"]["f_sampson_pct"] is None


def test_stereo_route_rejects_unbounded_request_parameters(monkeypatch):
    from app.routes import dual_camera

    monkeypatch.setattr(dual_camera, "run_stereo_liveness_check", lambda **kwargs: {"success": True})
    app = FastAPI()
    app.include_router(dual_camera.router)
    with TestClient(app) as client:
        for query in ("sample_count=-1", "sample_count=1000000", "warmup_seconds=-1", "warmup_seconds=inf"):
            assert client.post("/api/stereo-liveness/check?" + query).status_code == 422
