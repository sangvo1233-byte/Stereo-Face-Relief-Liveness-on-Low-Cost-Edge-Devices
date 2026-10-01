# Stereo Face-Relief Liveness on Low-Cost Edge Devices

Research code and numeric pilot evidence for **stereo face liveness detection**, **face anti-spoofing / presentation attack detection (PAD)**, and **Raspberry Pi 4 edge deployment** using two ordinary RGB cameras. The application also includes a face-recognition attendance dashboard built with FastAPI, InsightFace/ArcFace, MediaPipe, and SQLite.

This repository accompanies:

> Tieu Cong Son Long, Vo Minh Sang, Nguyen Minh Truong, Hoang Quoc Anh, and Nguyen Van Nam. **Toward Stereo Face-Relief Liveness on Low-Cost Edge Devices: An Algorithm Comparison and Feasibility Study.** 2026 5th International Conference on Electronics Representation and Algorithm (ICERA), pp. 580–585. IEEE. DOI: [10.1109/ICERA72709.2026.11666643](https://doi.org/10.1109/ICERA72709.2026.11666643).

[Paper and citation](paper/README.md) · [Reproduce pilot metrics](docs/reproduction.md) · [Architecture](docs/architecture.md) · [Setup and operations](docs/development.md)

Current repository-completion work is tracked in [the roadmap](docs/roadmap.md).

Publication metadata was checked against IEEE's Crossref deposit on 2026-10-01. Scopus indexing and ranking are not verified here. The current software is a documented working-copy snapshot; its identity with the exact revision used for the accepted paper's experiments has not been established.

## Start here

**Read the research:** open the [paper record](paper/README.md), then the [evidence and limitations](docs/reproduction.md). The contribution is a bounded comparison and feasibility protocol, using established stereo methods.

**Check the reported host pilot numbers:** Python 3.10+ is sufficient; no camera, model download, or application dependencies are needed.

```bash
git clone https://github.com/sangvo1233-byte/Stereo-Face-Relief-Liveness-on-Low-Cost-Edge-Devices.git Stereo-Face-Relief-Liveness-on-Low-Cost-Edge-Devices
cd Stereo-Face-Relief-Liveness-on-Low-Cost-Edge-Devices
python scripts/reproduce_pilot_metrics.py
```

**Run the camera demo:** follow [setup and operations](docs/development.md), then open `/dual-camera`. The stereo endpoint returns `LIVE`, `SPOOF`, or `INCONCLUSIVE`. It is an experimental path evaluated separately from the existing attendance scanner.

## What the research compares

| Method family | Implementation | Evidence boundary |
| --- | --- | --- |
| Sparse homography and H-vs-F geometry | `core/stereo_liveness.py`, `research/stereo_matching/17_epipolar_liveness.py` | Host pilot; short sparse endpoint Pi4 trial described in the paper |
| Temporal aggregation | `core/stereo_liveness.py` | Conservative voting and inconclusive decisions |
| Dense StereoSGBM | `research/stereo_matching/31_deep_stereo_pipeline.py` | Host smoke comparison; no dense Pi4 performance claim |
| HITNet / RAFT-Stereo | `research/stereo_matching/31_deep_stereo_pipeline.py`, `32_pi_three_options_trial.py` | Optional external ONNX weights; host comparison, not demonstrated Pi4 readiness |
| Synthetic geometry | `research/stereo_matching/34_synthetic_spoof_benchmark.py` | Geometry diagnostic, not a real presentation-attack dataset |

## Archived host pilot results

| Evidence | Total temporal windows | Included PAD windows | APCER | BPCER | ACER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary pilot | 592 | 296 | 0.00% | 8.05% | 4.03% |
| Second-participant extension | 311 | 204 | 0.00% | 30.88% | 15.44% |
| Combined pilot summary | 903 | 500 | 0.00% | 15.21% | 7.60% |

These are two-participant, single-rig, **screen-replay-only host measurements**. `EMPTY` and `PARTIAL` windows are excluded from PAD error rates. `INCONCLUSIVE` counts as a rejection for bona fide windows and as non-live for attacks. The combined summary contains 217 bona fide and 283 screen-spoof windows. In the extension, 132 of 136 attack windows were inconclusive; non-acceptance is not the same as explicit spoof classification.

The paper separately reports a short Pi4 sparse endpoint trial: 9/10 live-user checks returned `LIVE`, and 10/10 screen-replay checks returned `SPOOF`. Its raw runtime archive is not included in this candidate; these values are paper-reported rather than independently reproduced here. Print attacks, broad PAD generalization, and sustained edge stability remain unverified.

## Repository map

```text
paper/                         Published paper record and BibTeX
results/pilot/                 Numeric CSV and JSON host evidence; no face media
research/stereo_matching/      Selected research scripts and their dependencies
core/                         Camera, stereo geometry, attendance runtime, storage
app/routes/                   REST and WebSocket APIs
web/                          Attendance dashboard and dual-camera demo
scripts/                      Model download, dataset recorder, benchmark, metric check
tests/                        Existing runtime and stereo tests
docs/                         Architecture, reproduction, development, source provenance
Dockerfile, docker-compose.yml Raspberry Pi deployment source
models/, database/, logs/      Downloaded/private mutable runtime state; excluded from Git
```

Legacy V1 and V3 attendance routes remain alongside V4.4. [Architecture](docs/architecture.md) explains which paths own each flow. `docs/15-realignment-plan.md` is a historical migration record, not the current task list.

## Validation and reuse

The archived numeric check can run independently. The application tests require the dependencies and should be run with `python -m pytest -m "not integration" -q`; camera/Pi smoke tests are separate. See [development commands](docs/development.md) for the precise verification boundaries.

Face photos, paired videos, embeddings, databases, secrets, and model weights are not distributed. Full inference reproduction requires your own authorized captures, model artifacts, and recorded rig configuration. The paper and third-party models have separate rights; a code license has not yet been selected. See [reuse and publication](docs/reproduction.md#reuse-and-publication).

Vietnamese: mã nghiên cứu kiểm tra tính sống khuôn mặt bằng hai camera RGB giá rẻ, chống giả mạo khuôn mặt và thử nghiệm trên Raspberry Pi 4. Bắt đầu bằng [paper](paper/README.md) hoặc [hướng dẫn kiểm tra số liệu](docs/reproduction.md).
