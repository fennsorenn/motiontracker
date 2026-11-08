# train (implemented; M1 itself WIP)

Location: `train/` (uv project, package `mttrain`, CUDA PyTorch). Builds M1 training data from CMU, trains the body lifter and runs it on recorded sessions. Depends on `mtcore` (`modules/mtcore.md`); its output is evaluated with the harness (`modules/eval-harness.md`). Plan, decisions and run results: `plans/m1-body-lifter.md`.

All CMU-derived data and checkpoints live in gitignored `data/` (`data/m1/...`) and are never committed.

## Pipeline (in order)
| Module | Role |
|---|---|
| `prep_fits` | Batch CMU → MHR fit (`mtcore.cmu_fit`) into `data/m1/<dir>` (`--out`); stores per-subject `scales`, `head_offset`, `head_post` in `subjects.json`; QC flags bad clips (arm residual, wrist speed) for exclusion. |
| `seated` | Standing → seated: seated legs, damped pelvis lean/turn, breathing/sway, per-arm rest postures (lap/desk/hanging), clamped to MHR limits. `rest_weights` no longer fades rest stretches out at the crop end (fixed 2026-11-08). |
| `camera` | Desk-webcam camera (height up to 0.55 m), framing sampled around the recorded session. vspark mirrors the image, so labels match an unmirrored synthetic camera. World landmarks are camera-aligned, hip-centred, metres. |
| `mpstyle` | MediaPipe-style landmark placement: biases applied in both spaces vs world-only distortions, calibrated against the fitted CMU population, randomized per sequence. `IMAGE_Z = 3.0` scales image-landmark depth, with a log-normal spread. |
| `noise` | Observation model: AR(1) jitter, visibility by distance outside the frame (with lag), drifting guesses, glitches, dropouts, lower visibility for seated hips. Per-group recalibration `_RECAL` (`Z_GAIN`, `IMAGE_GAIN`); `NoiseConfig.image_z_noise`. |
| `synth` | Ties the above together: fitted MHR sequence → synthetic landmark frames (world + image + visibility) at the target frame times. |
| `features` | Shared by training and inference: causal shoulder-width scale, history at time offsets 0–1.6 s. |
| `skeleton` | Target extraction from MHR poses and torch FK for the position loss. |
| `dataset` | Numpy `Generator`: on-the-fly samples; 75% seated. Validation = held-out CMU subjects (fixed numpy set, also with `--gen gpu`). `MT_FITS` selects the fits directory under `data/m1` (default `fits`; current `fits-v3`), recorded in the run config. |
| `model` | `Lifter`: 17 rotations (6D) + per-joint confidence, ~1.1M params, ~0.3 ms/frame CPU. |
| `losses` | Laplace NLL on rotation (uses the confidence), FK position, velocity, joint-limit box measured from the data (proxy for MHR's Euler limits). |
| `train` | Training loop; runs in `data/m1/runs/<name>`. `--gen gpu` (default) uses `gpugen` on `--device` (default `cuda:1`), `--gen-chunk 4` batches per generator call; `--gen cpu` keeps the DataLoader path. |
| `infer` | Session → harness pose track; writes only the VRM bones the model drives. |
| `calibrate` | Real vs synthetic landmark statistics. Rerun after new sessions are recorded. |
| `limitcheck` | Criterion 8: `python -m mttrain.limitcheck <ckpt> <session>`. IK of MHR parameters to the predicted orientations with limits off, checked against `[Limits]` (angles wrapped). Runs on the session plus stressed copies (tracking loss, low confidence, extra noise, arms out of frame, glitches) and on synthetic ground truth for validation. |

## GPU data generator (`gpugen`, implemented 2026-11-08)
Package `mttrain/gpugen/`: batched torch port of the synthesis pipeline; the training default. Builds whole batches on the RTX 4060 instead of CPU DataLoader workers, which crashed the machine (see ARCHITECTURE key decisions). ~4,800 samples/s, 40k steps in 36 min, ~1 CPU core.

| File | Role |
|---|---|
| `fk.py` | `MhrFK`: MHR forward kinematics in torch. Optional `keep` joint list: only those joints and their ancestors are computed (the generator uses the 44 joints landmarks and targets need). |
| `geom.py` | `Landmarks`, `Style` (batched MediaPipe biases), `sample_desk_camera`, `project`, `world_landmarks`, `Targets`. |
| `motion.py` | `Seater` (seated motion + limit clamp), `rest_weights`, `slow_signals`. |
| `observe.py` | `sample_config`, `observe` (noise, visibility). |
| `features.py` | `causal_scale`, `frame_features`, `history_indices`, `model_input`; exact port of `mttrain.features`. |
| `generator.py` | `GpuGenerator(split, device, K, p_seated, seed).sample(B)`: same dict as `dataset.Generator.sample`, batched. |

Differences from the numpy generator: fixed 97-frame source crop (clips shorter than 97 frames are unused) and different random streams. Tests (`tests/test_gpugen.py`): FK vs pymomentum; geometry and features exact vs numpy; the whole generator statistically vs numpy.

Extending: any change to a numpy synthesis stage must be mirrored in `gpugen` (and vice versa), with the comparison test kept passing; the numpy pipeline is the reference.

## Running heavy jobs
Run CPU-heavy jobs (fits, training) through `tools/guarded-run.sh [-c CORES] [-l MAX_LOAD] [-m MIN_AVAIL_GB] [-o LOG] -- command...`. It pins the job and all its threads to a core set (default: upper half of the logical CPUs, i.e. 8-15 here), nices it, and kills the whole process group if available memory drops below 6 GB or the load runs away. Pinning is needed because pymomentum's dispenso thread pool starts one thread per logical CPU in every process regardless of OMP/BLAS limits or affinity; `taskset` still confines where those threads run, so it is the only reliable cap. The `fits-v3` refit (16 min, 3 workers) ran under it.

## Extending
- Anything inference needs (feature construction, scaling) must stay in `features` so training and inference cannot diverge.
- New observation effects belong in `noise` (sensor-like) or `mpstyle` (systematic MediaPipe placement); check them with `calibrate` against real sessions.

## Open
Arm rest noise (not yet better than the baseline), fairness of criterion 7 and the go/no-go on seated-v2 sessions: see the plan (step 7b).
