# Stereo research scripts

Selected implementations supporting the method families discussed in the ICERA paper. These are research tools; use [reproduction](../../docs/reproduction.md) for evidence boundaries and [setup](../../docs/development.md) for application dependencies. Plots additionally require Matplotlib.

| Script | Purpose | Input / output |
| --- | --- | --- |
| `03_liveness_planar.py`, `11_rig_sweep.py` | Planarity primitive and simulated rig geometry | Helpers used by the temporal/jitter experiments |
| `13_temporal_average.py` | Temporal noise/geometry simulation | Analytic/synthetic results, not PAD measurements |
| `15_measure_jitter.py` | Landmark sensitivity to image variants | Your face image and landmark model; generated plots |
| `17_epipolar_liveness.py` | Sparse H-vs-F research prototype | Explicit `--left` and `--right` image paths |
| `30_rig_geometry_calculator.py` | Baseline/relief design estimates | Geometry arguments; analytic results |
| `31_deep_stereo_pipeline.py` | SGBM or optional ONNX disparity runner | Image pair, model/preprocessing if ONNX, output directory |
| `32_pi_three_options_trial.py` | Comparison wrapper around the dense runner | Pair plus separately obtained HITNet/RAFT weights; runtime is on the machine executing it |
| `34_synthetic_spoof_benchmark.py` | Synthetic planar/curved diagnostic | Your left image and selected backend |

Run each tool with `--help`. Supply your own authorized inputs; old image paths in historical script examples are not distributed inputs. The dense wrapper's filename does not establish measurement on a Pi. The live application uses `core/stereo_liveness.py`, not these numbered scripts.

Example CPU smoke invocation after setup:

```bash
python research/stereo_matching/31_deep_stereo_pipeline.py --left datasets/example/left.png --right datasets/example/right.png --backend sgbm --out logs/stereo_trials/sgbm
```

Use appropriate calibrated/rectified inputs before interpreting disparity as metric depth. A successful run on new images is not a reproduction of the paper's original host timings or PAD result.
