# train (implemented; M1 itself WIP)

Location: `train/` (uv project, package `mttrain`, CUDA PyTorch). Builds M1 training data from CMU, trains the body lifter and runs it on recorded sessions. Depends on `mtcore` (`modules/mtcore.md`); its output is evaluated with the harness (`modules/eval-harness.md`). Plan, decisions and run results: `plans/m1-body-lifter.md`.

All CMU-derived data and checkpoints live in gitignored `data/` (`data/m1/...`) and are never committed.

## Pipeline (in order)
| Module | Role |
|---|---|
| `prep_fits` | Batch CMU → MHR fit (`mtcore.cmu_fit`) into `data/m1/<dir>` (`--out`); stores per-subject `scales`, `head_offset`, `head_post` in `subjects.json`; QC flags bad clips (arm residual, wrist speed) for exclusion. |
| `seated` | Standing → seated: seated legs, damped pelvis lean/turn, breathing/sway, per-arm rest postures (lap/desk/hanging), clamped to MHR limits. `rest_weights` no longer fades rest stretches out at the crop end (fixed 2026-11-08). |
| `camera` | Desk-webcam camera (height up to 0.55 m). Sampled widely (commit d437e72): `FOV_RANGE` 50-130 deg horizontal, `DIST_RANGE` 0.45-1.6 m camera to shoulders (log-uniform), clamped so the shoulder width covers `SW_RANGE` 10-50% of the frame; shoulders at 40-75% of the frame height. `sample_desk_camera(..., fov=)` overrides the FOV with the same random stream: camera stays on the same line from the chest (height scaled), same framing. `shoulder_distance(cam, X)` feeds the depth flattening. vspark mirrors the image, so labels match an unmirrored synthetic camera. World landmarks are camera-aligned, hip-centred, metres. |
| `mpstyle` | MediaPipe-style landmark placement: biases applied in both spaces vs world-only distortions, calibrated against the fitted CMU population, randomized per sequence. `IMAGE_Z = 3.0` scales image-landmark depth, with a log-normal spread. Depth flattening: see below. |
| `noise` | Observation model: AR(1) jitter, visibility by distance outside the frame (with lag), drifting guesses, glitches, dropouts, lower visibility for seated hips. Per-group recalibration `_RECAL` (`Z_GAIN`, `IMAGE_GAIN`); `NoiseConfig.image_z_noise`. Coherent arm noise (2026-11-09): `MIX`, a 33x33 mixing matrix built from `ARM_CORR` (shoulder-elbow 0.32, elbow-wrist 0.52, wrist-hand 0.99), correlates the jitter and the out-of-frame guesses along each arm; mirrored in `gpugen/observe.py`. |
| `synth` | Ties the above together: fitted MHR sequence → synthetic landmark frames (world + image + visibility) at the target frame times. Passes the shoulder distance to `Style.flatten`. |
| `features` | Shared by training and inference: causal shoulder-width scale, history at time offsets 0–1.6 s. |
| `skeleton` | Target extraction from MHR poses and torch FK for the position loss. `CONTACT_POINTS` and `contact_points(G, P, cof)` (see Contacts). |
| `dataset` | Numpy `Generator`: on-the-fly samples; 75% seated. Validation = held-out CMU subjects (fixed numpy set, also with `--gen gpu`). `MT_FITS` selects the fits directory under `data/m1` (default `fits`; current `fits-v3`), recorded in the run config. `source_table(split, p_choreo)` mixes in the choreography pool (see Contacts). |
| `model` | `Lifter`: 17 rotations (6D) + per-joint confidence, ~1.1M params, ~0.3 ms/frame CPU. |
| `losses` | Laplace NLL on rotation (uses the confidence), FK position, velocity, joint-limit box measured from the data (proxy for MHR's Euler limits), contact loss `con` (see Contacts). |
| `train` | Training loop; runs in `data/m1/runs/<name>`. `--gen gpu` (default) uses `gpugen` on `--device` (default `cuda:1`), `--gen-chunk 4` batches per generator call; `--gen cpu` keeps the DataLoader path. |
| `infer` | Session → harness pose track; writes only the VRM bones the model drives. |
| `calibrate` | Real vs synthetic landmark statistics. Rerun after new sessions are recorded. |
| `render` | Rendered synthetic sessions with exact ground truth (see below). |
| `choreo` | Designed contact choreography (see Contacts). |
| `limitcheck` | Criterion 8: `python -m mttrain.limitcheck <ckpt> <session>`. IK of MHR parameters to the predicted orientations with limits off, checked against `[Limits]` (angles wrapped). Runs on the session plus stressed copies (tracking loss, low confidence, extra noise, arms out of frame, glitches) and on synthetic ground truth for validation. |

## GPU data generator (`gpugen`, implemented 2026-11-08)
Package `mttrain/gpugen/`: batched torch port of the synthesis pipeline; the training default. Builds whole batches on the RTX 4060 instead of CPU DataLoader workers, which crashed the machine (see ARCHITECTURE key decisions). ~4,800 samples/s, 40k steps in 36 min, ~1 CPU core.

| File | Role |
|---|---|
| `fk.py` | `MhrFK`: MHR forward kinematics in torch. Optional `keep` joint list: only those joints and their ancestors are computed (the generator uses the 44 joints landmarks and targets need). |
| `geom.py` | `Landmarks`, `Style` (batched MediaPipe biases; `depth_at`, `flatten(W, dist)`), `sample_desk_camera`, `shoulder_distance`, `project`, `world_landmarks`, `Targets`. |
| `motion.py` | `Seater` (seated motion + limit clamp), `rest_weights`, `slow_signals`. |
| `observe.py` | `sample_config`, `observe` (noise, visibility). |
| `features.py` | `causal_scale`, `frame_features`, `history_indices`, `model_input`; exact port of `mttrain.features`. |
| `generator.py` | `GpuGenerator(split, device, K, p_seated, seed).sample(B)`: same dict as `dataset.Generator.sample`, batched. Flattening uses the shoulder distance over valid frames. |

Differences from the numpy generator: fixed 97-frame source crop (clips shorter than 97 frames are unused) and different random streams. Tests (`tests/test_gpugen.py`): FK vs pymomentum; geometry and features exact vs numpy; the whole generator statistically vs numpy.

Extending: any change to a numpy synthesis stage must be mirrored in `gpugen` (and vice versa), with the comparison test kept passing; the numpy pipeline is the reference.

## Rendered sessions (`render`, implemented 2026-11-09)
`python -m mttrain.render prepare|video|replay --out data/m1/renders/<set> [--n] [--seconds] [--seed] [--rest] [--fovs 50,65,80,95]`, with `tools/render/render_blender.py` (Blender 5.2 EEVEE, `blender -b`). Pipeline:
1. `prepare`: held-out CMU clips (distinct clips) through the seated pipeline at 30 fps; camera from the training desk-camera sampler; writes `gt.jsonl` (camera-relative ground-truth VRM track).
2. Blender renders frames: MHR mesh skinned with pymomentum, plain skin material, wall, key light.
3. `video`: ffmpeg → mirrored VP9 video.
4. `replay`: recorder replay in headless Chrome (`tools/recorder/test/drive_headless.mjs`, optional timeout argument) → landmarks from the exact live MediaPipe build (`modules/recorder.md`).

`--contact`: contact choreography instead of CMU clips (see Contacts). `--rest`: held posture with both arms in rest templates (`seated.seat(force_rest=...)`). `--fovs` (commit f666bf2): each sequence written once per horizontal FOV as `seqNN_fovFF`, same motion, appearance and camera line, only the perspective changes. Evaluate with `mteval evaluate --gt gt.jsonl` (`modules/eval-harness.md`). Heavy steps run under `tools/guarded-run.sh`. Renders are CMU-derived: `data/m1/renders`, never committed. Sets so far: `v1` (motion, 8 x 20 s), `rest1` (6 x 15 s), `fov1` (3 clips x FOV 50/65/80/95), `contact1` (contact choreography); results in the plan, steps 7c-7d.

Known limits: the seated idle motion is a rough AR(1), so rest ground truth is never still (smoothing it is an open owner decision). Plain mannequin renders flatten depth more than real video does (see below).

## MediaPipe depth flattening (implemented, M1, commits 1913fdb, d437e72)
Owner decision (2026-11-09): train the model to undo the flattening ("train it to flatten less").

Rendered sessions (exact ground truth) showed MediaPipe's world landmarks compress depth per segment: depth component of a segment vector ≈ gain × true + bias, with segment length kept (the image-plane part is stretched). Measured: upper arm 0.37 / +1 cm, forearm 0.33 / −6 cm, hand 0.18 / −4 cm, shoulders 0.60 / +1 cm, hips 0.47 / −1 cm, torso 0.42 / −10 cm, head yaw (ears) 0.47, ears→nose 0.43 / −5 cm. Image z ≈ 3× world z on every segment. The previous synthesis had unflattened world depth and image z = 3× geometric depth, i.e. ~3× too deep in both spaces, so models inherited the flattening on real input. (vspark compensates with `ikCalibration.zScale = 3`, only in its IK mode, off by default; the frozen baseline does not use it.)

Implementation (`mpstyle`):
- `DEPTH` (gain/bias table) and `DEPTH_STEPS`: hierarchical steps (torso, shoulders, hips, neck, ears, face points, upper arm, forearm, hand points); each step translates only the points below it.
- `flatten()`: depth = gain · dz + bias, clipped to ±segment length; image-plane part stretched to keep the length, stretch capped at `FLAT_MAX_STRETCH = 4`.
- `Style.depth_gain` / `Style.depth_bias`, randomized per sequence (gains log-normal 0.25, clipped to [0.05, 1]; bias + N(0, 1 cm + 30% |bias|)). `DEPTH` holds gains at `DEPTH_REF_DIST = 1.1` m (camera to shoulders, median of the calibration renders).
- Perspective: `DEPTH_SLOPE` (per segment, gain change per metre of camera distance, from the FOV experiment: closer cameras flatten more).
- `Style.depth_relax`, uniform 0-1 per sequence: 0 = render-calibrated flattening, 1 = none. Real video flattens less than the renders, so the model has to infer the strength.
- `Style.depth_at(dist)`: gain with the slope applied (clipped to [0.02, 1]), then blended toward 1 (and bias toward 0) by `depth_relax`. `Style.flatten(W, dist)` uses it; `synth` passes `camera.shoulder_distance`.
- `Camera.flat_image_z`: image-landmark z from the flattened depths.
- `synth`: world = `style.world(style.flatten(world_landmarks))`; image z from flattened depth, then × `IMAGE_Z`.
- `gpugen`: `geom.Style.depth_at` / `flatten(W, dist)`, `shoulder_distance`, `flat_image_z`, used in `generator.py` (shoulder distance over valid frames); tests check exact numpy/torch agreement.

Results (plan step 7d): `m1-v6` (render-strength flattening only) is much better on renders but worse on the real session (2D wrist 6.0 → 9.1 cm): on real video MediaPipe's forearm depth matches the image foreshortening, so real flattening is weaker than on the plain renders. `m1-v7` (wide cameras, distance slope, relax factor) is the current best overall: real session 2D wrist 6.7 cm vs baseline-rendered 7.2, renders close to `m1-v6`.

Known limits / not modelled:
- Synthetic noise does not grow when the person is small in the frame (now down to 10% shoulder width).
- Pinhole projection only; wide lenses add barrel distortion.
- The extra depth error during motion (4-7 cm residual, ~0.5 s timescale; <1.5 cm at rest).
- Real-video flattening strength is unmeasured; it needs multi-view ground truth (owner is setting up two cameras).

## Contacts (choreography and contact-aware training implemented)
Motivation: on `contact1` every system leaves the hand 11-14 cm too far from the face; palms together `m1-v7` is best (3.9 cm |err|).

Choreography (`choreo`, implemented, commits 546e137, 8fffdcf):
- Seated sequences with designed contact events, solved with MHR IK (pymomentum `solve_ik`): prayer palms, hand-to-face with fingertip or palm on chin/cheek/forehead/mouth, and near-misses at 8-14 cm. Minimum-jerk approach / hold / release; the exact contact intervals are the holds.
- Rendered test set: `render.py prepare --contact` → `data/m1/renders/contact1`; segments have `settleMs` 0 and `trueCm`. The harness pools all segments of a variant and shows a ground-truth contact column (`modules/eval-harness.md`).
- Training pool (commit 0ed4a2b): `MT_FITS=fits-v3 uv run python -m mttrain.choreo --split train --n 1000` (and `--split val --n 100 --seed 101`) → `data/m1/choreo/<split>/cNNNN.npz` (MediaPipe landmarks at 30 fps + segments JSON). CMU-derived, local only. <1 s per 30 s sequence; run under `tools/guarded-run.sh`.

Contact-aware training (implemented, commit 0ed4a2b; first model `m1-v8`, 40k steps on GPU):
- `dataset.source_table(split, p_choreo)`: the pool is a training source with `p_choreo` of the total weight (default 0.2, `train.py --p-choreo`). Pool sequences are already seated and skip the seater, in both the numpy `Generator` and `gpugen.GpuGenerator`.
- `skeleton.CONTACT_POINTS`: `l_hand` / `r_hand` = midpoint of the index and pinky knuckle landmarks (parent wrist), `nose` (parent `c_head`). Stored as offsets `cof` in the parent's rest-relative frame (`TargetExtractor.contact_offsets`, `gpugen` `Targets.contact_offsets`); `contact_points(G, P, cof)` places them.
- `losses` `con` (weight 5): vector error between contact-point differences (hand-hand, each hand-nose), after FK on the true body, weighted `exp(-(d_true / 0.15 m)^2)`. Validation metrics `cm_contact_near` / `frac_contact_near` (pairs closer than 15 cm).
- Extending: a new contact point goes into `CONTACT_POINTS` with a parent joint, and both target extractors (numpy and `gpugen`) must produce its offset.

Results (`m1-v8` vs `m1-v7`, plan step 7e; full numbers there):
- Synthetic validation, near pairs: 3.5 cm (7.3).
- `contact1`, 7b on MediaPipe's skeleton: hand-to-face 7.9 cm (11.1); palms 4.9 cm (3.9), now too close on that skeleton. On the true body: palms 6.8 cm (10.0), hand-to-face 25.2 cm (29.7).
- Elsewhere (motion/FOV/rest renders, real session) about level with `m1-v7`.

Open: hand-to-face is a MediaPipe depth problem. During true contacts MediaPipe places the hand 8-26 cm in front of the nose (world z), the same as for real near-misses; the synthetic observation model (`mpstyle`/`noise`) has no such bias, so every model puts the hand ~30 cm toward the camera. Unverified on real video: needs seated-v2 recordings with contact segments.

## Running heavy jobs
Run CPU-heavy jobs (fits, training) through `tools/guarded-run.sh [-c CORES] [-l MAX_LOAD] [-m MIN_AVAIL_GB] [-o LOG] -- command...`. It pins the job and all its threads to a core set (default: upper half of the logical CPUs, i.e. 8-15 here), nices it, and kills the whole process group if available memory drops below 6 GB or the load runs away. Pinning is needed because pymomentum's dispenso thread pool starts one thread per logical CPU in every process regardless of OMP/BLAS limits or affinity; `taskset` still confines where those threads run, so it is the only reliable cap. The `fits-v3` refit (16 min, 3 workers) ran under it.

## Extending
- Anything inference needs (feature construction, scaling) must stay in `features` so training and inference cannot diverge.
- New observation effects belong in `noise` (sensor-like) or `mpstyle` (systematic MediaPipe placement); check them with `calibrate` against real sessions.

## Open
Current best run `m1-v7` (depth flattening; `m1-v5`, `m1-v6` kept); contact-aware `m1-v8` level elsewhere, better at contacts. The MediaPipe hand-to-face depth bias (Contacts). Arm rest noise (not yet better than the baseline), the rough seated idle motion, real flattening strength (multi-view) and the go/no-go on seated-v2 sessions: see the plan (steps 7b-7e) and the depth flattening limits above.
