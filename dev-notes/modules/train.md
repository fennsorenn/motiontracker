# train (implemented; M1 itself WIP)

Location: `train/` (uv project, package `mttrain`, CUDA PyTorch). Builds M1 training data from CMU, trains the body lifter and runs it on recorded sessions. Depends on `mtcore` (`modules/mtcore.md`); its output is evaluated with the harness (`modules/eval-harness.md`). Plan, decisions and run results: `plans/m1-body-lifter.md`.

All CMU-derived data and checkpoints live in gitignored `data/` (`data/m1/...`) and are never committed.

## Pipeline (in order)
| Module | Role |
|---|---|
| `prep_fits` | Batch CMU → MHR fit (`mtcore.cmu_fit`) into `data/m1/fits/`; QC flags bad clips (arm residual, wrist speed) for exclusion. |
| `seated` | Standing → seated: seated legs, damped pelvis lean/turn, breathing/sway, per-arm rest postures (lap/desk/hanging), clamped to MHR limits. |
| `camera` | Desk-webcam camera, framing sampled around the recorded session. vspark mirrors the image, so labels match an unmirrored synthetic camera. World landmarks are camera-aligned, hip-centred, metres. |
| `mpstyle` | MediaPipe-style landmark placement: biases applied in both spaces vs world-only distortions, calibrated against the fitted CMU population, randomized per sequence. |
| `noise` | Observation model: AR(1) jitter, visibility by distance outside the frame (with lag), drifting guesses, glitches, dropouts, lower visibility for seated hips. |
| `synth` | Ties the above together: fitted MHR sequence → synthetic landmark frames (world + image + visibility) at the target frame times. |
| `features` | Shared by training and inference: causal shoulder-width scale, history at time offsets 0–1.6 s. |
| `skeleton` | Target extraction from MHR poses and torch FK for the position loss. |
| `dataset` | On-the-fly samples; 75% seated. Validation = held-out CMU subjects. |
| `model` | `Lifter`: 17 rotations (6D) + per-joint confidence, ~1.1M params, ~0.3 ms/frame CPU. |
| `losses` | Laplace NLL on rotation (uses the confidence), FK position, velocity, joint-limit box measured from the data (proxy for MHR's Euler limits). |
| `train` | Training loop; runs in `data/m1/runs/<name>`. |
| `infer` | Session → harness pose track; writes only the VRM bones the model drives. |
| `calibrate` | Real vs synthetic landmark statistics. Rerun after new sessions are recorded. |

## Extending
- Anything inference needs (feature construction, scaling) must stay in `features` so training and inference cannot diverge.
- New observation effects belong in `noise` (sensor-like) or `mpstyle` (systematic MediaPipe placement); check them with `calibrate` against real sessions.

## Open
Render stage for model output (how `baseline-rendered` is applied) and the go/no-go on seated-v2 sessions: see the plan.
