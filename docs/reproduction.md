# Evidence and reproduction

## Recheck the archived host metrics

```bash
python scripts/reproduce_pilot_metrics.py
```

This standard-library-only command counts nonempty `window_verdict` values in the archived CSVs, checks them against the original summary JSONs, and prints each pilot plus the combined summary. It does not load models, read images, create files, or rerun inference.

| Run | Sampled pairs | Total windows | Bona fide windows | Screen-spoof windows |
| --- | ---: | ---: | ---: | ---: |
| `20260714_160804` | 600 | 592 | 149 | 147 |
| `20260718_001027` | 325 | 311 | 68 | 136 |
| Combined | 925 | 903 | 217 | 283 |

The primary pilot is one participant; the extension adds a second participant on the same rig. Both are host-side evidence. The total includes `EMPTY` and `PARTIAL`; the PAD denominator excludes those labels. Windows overlap, so 903 windows are not 903 independent subjects or presentations.

For these included classes:

- APCER = `SCREEN_SPOOF -> LIVE` / all screen-spoof windows.
- BPCER = (`LIVE_USER -> SPOOF` + `LIVE_USER -> INCONCLUSIVE`) / all bona fide windows.
- ACER = (APCER + BPCER) / 2.

The primary pilot has 12/149 rejected bona fide windows; the extension has 21/68. The combined BPCER is 33/217 = 15.21%. No included screen-spoof window is `LIVE`, giving 0/283 observed attack acceptance. The extension has 4 explicit `SPOOF` and 132 `INCONCLUSIVE` attack windows. Preserve that distinction when describing the result.

## Rerun inference on your own captures

Install the [application environment](development.md), obtain the landmark model, and record a fixed rig with the existing recorder:

```bash
python scripts/stereo_dataset_recorder_tkinter.py --help
python scripts/stereo_dataset_benchmark.py --help
python scripts/stereo_dataset_benchmark.py --dataset datasets/stereo_liveness --output output/stereo_eval --stride 5 --window 3
```

The recorder requires a desktop/Tkinter environment and two cameras. Each clip directory must contain `metadata.json`, `left.avi`, and `right.avi` as produced by the recorder. Labels used in the paper are `LIVE_USER`, `SCREEN_SPOOF`, `EMPTY`, and `PARTIAL`. A larger experiment may add attack labels but must keep its protocol and results separate from the published pilot.

The benchmark compares capture alignment metadata with current configuration and skips mismatched clips by default. Record the camera model, sources, baseline, capture resolution, alignment settings, host timestamp skew, lighting, participant and session pseudonyms, software revision, model hashes, stride, window, and threshold settings. The public working-copy snapshot is not proven to be the exact revision used for the archived runs; running it today need not reproduce the original verdicts.

## What is and is not independently reproducible here

| Paper evidence | Available in this candidate | Reproduction limit |
| --- | --- | --- |
| Primary and extension host metrics | Original numeric CSV/JSON and a checked recalculation command | Arithmetic/count reproduction; source video stays private |
| Sparse temporal demo | Source, recorder, benchmark, existing tests | Needs models, two cameras or authorized clips, and rig parameters |
| Dense/deep comparison | Research runners | Needs original image pairs, optional ONNX weights, exact preprocessing and host environment |
| Calibration/rectification | Numeric values described by the paper | Original board captures/calibration archive not included |
| Short Pi4 endpoint trial | Described by the paper | Raw target-device archive not included; not rerun in this preparation |
| Accepted manuscript | DOI, publisher metadata, citation | Accepted/camera-ready file identity still requires owner confirmation |

The paper reports a two-C270 640x480 MJPG input rig, 640x452 aligned output, approximately 12–13 cm baseline, and a separate 40-pair checkerboard calibration. Those settings are provenance from the study, not universal camera configuration.

Do not claim print-spoof performance, held-out general PAD accuracy, dense/deep Pi4 readiness, long-duration thermal/RAM stability, or hardware exposure synchronization from these artifacts. Acceptance/publication does not make those extra claims supported.

## Reuse and publication

Code licensing has not yet been selected by the owners; public visibility alone is not an open-source license grant. Paper rights and third-party model rights are separate from code rights. In particular, check the [InsightFace code/model licensing distinction](https://github.com/deepinsight/insightface#license) before reusing its weights.

Use the [DOI paper record](../paper/README.md) for the publication. A public manuscript copy must be the identified permitted version with the appropriate rights notice and posting location under [IEEE's conference posting policy](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/post-publication-policies/). No full manuscript PDF, LaTeX text, manuscript figures, face images/videos, real embeddings, database, secret, or model weight was added in this candidate.
