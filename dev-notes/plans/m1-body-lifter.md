# Plan: M1, body keypoint lifter POC in MHR space

> Branch: `feature/m1-body-lifter` · Status: draft
> This plan is the seed context for a cloud worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

Train the first learned model: MediaPipe pose landmarks → small causal temporal model → MHR
upper-body rotations, converted to VRM 1.0. Compare it with the baseline on real seated sessions
using the provisional success criteria (PLAN.md §6). This is the go/no-go point for the learned
approach.

## Constraints

- Strictly causal, real-time, lightweight (< 2 ms per frame on a laptop CPU) `[USER]`.
- MHR is the canonical skeleton; output goes through the spike's MHR → VRM conversion `[USER]`.
  Performer identity stays within MHR's human scale limits; flexible translations stay zero;
  forearm twist onto lowerArm via swing-twist (PLAN.md §3.10).
- Permissive licensing only. CMU data and anything derived from it (fits, training sets) live in
  gitignored `data/` and are never committed. Trained weights derived from CMU are fine to keep
  locally; whether to publish them is a later licensing question (register entry needed).
- Body bones only: spine, neck, head, shoulders, arms. Hands/fingers, face, merger, retargeting,
  root translation are later milestones.

## Decisions (owner, 2026-11-01)

1. **Input synthesis:** MediaPipe-style pose landmarks are computed directly from fitted MHR poses
   (each of the 33 landmarks from an MHR joint or mesh point), plus a noise model calibrated on
   real sessions and simulated camera framing (landmarks leaving the frame get low visibility and
   MediaPipe-like drifting guesses). Render-then-detect stays in M6.
2. **Model:** small causal model reading past samples at fixed time offsets (≈ 0.1–1.6 s, sampled
   by time, not frame count, with explicit differences). Output 6D rotations for the MHR spine,
   neck, head, clavicles, upper arms, forearms and wrists. Losses: rotation, elbow/wrist position
   via FK, joint-limit penalty, velocity. Preferred over a GRU (cleaner with dropped frames,
   stateless per call, simple ONNX export).
3. **Data:** CMU clips with little root travel (≈ 760 of 2514, `scan_cmu.py`), weighted toward
   upper-body activity; full-body clips stay in. Seated motion: CMU upper bodies on a seated
   lower body plus idle sway. Performer proportions: each subject's fit, randomly varied within
   MHR's limits. Procedural reaching (minimum-jerk) later.
4. **Code layout:** new `train/` uv project with CUDA PyTorch. Spike code that M1 reuses (CMU
   reader, CMU → MHR fit, MHR → VRM conversion, MHR loading) moves into a shared package instead
   of being imported from `spikes/`.
5. **Shared package:** `core/`, Python package `mtcore` (follow-up, 2026-11-01).
6. **Per-bone confidence** is output from the start, trained with an uncertainty-weighted loss
   (follow-up, 2026-11-01; PLAN.md §3.4 needs it for the M4 merger).
7. **Spine:** the model predicts all four MHR spine joints (`c_spine0..3`); the conversion folds
   them onto VRM's spine bones (follow-up, 2026-11-01).

## Files in scope

- `core/` (`mtcore`) — shared package: MHR loading/FK, CMU reader + fit, MHR → VRM conversion,
  moved from `spikes/m3-mhr-vrm/` with tests. The spike keeps working or is pointed at it.
- `train/` — uv project (CUDA PyTorch): data prep scripts, dataset, model, training loop, export.
- `data/m1/` (gitignored) — fitted CMU sequences, generated training sets, checkpoints.
- `eval/` — run the trained model on recorded sessions as a system (`name=path.jsonl` already
  works; a built-in system can follow).
- `docs/LICENSES-REGISTER.md` — entries for derived data and checkpoints.

## Out of scope

- Hands, face, merger, retargeting layer (M4); root translation and calibration (M5); rendering
  (M6); feature taps (M7); knobs (M8); the service (M2).

## Approach

1. **Shared package.** (done) Move the reused spike modules, add tests for the conversion round trip
   and the CMU fit (small, fast).
2. **Batch CMU → MHR fit.** (done 2026-11-01: 760 clips, 3.1 h at 30 fps, arm residual median 0.4 cm;
   QC flags 23 clips with arm p95 > 3 cm and 18 with wrist speeds > 10 m/s, to exclude.) Fit the selected clips at 30 fps (`step=4` on 120 fps), store MHR
   model parameters + subject scales per clip in `data/m1/fits/`. Log residuals; drop clips with
   bad fits. Loosen the spine pairing (spike finding: spine scales lean on their limits).
   About 0.3 s per 100 frames, so the whole set takes minutes to an hour on CPU.
3. **Landmark synthesis.** (done: `mtcore.landmarks`, `mttrain.camera`, `mttrain.synth`.) Map MediaPipe's 33 pose landmarks to MHR joints/mesh points (face
   points like nose, eyes, ears, mouth corners from the mesh; hips/shoulders/elbows/wrists from
   joints with an offset). Produce both outputs the recorder stores: world landmarks (metres,
   hip-centred, MediaPipe axes) and image landmarks (`pose2d`, from a sampled camera: desk-webcam
   framing, FOV, distance, tilt). Validate against the real session: bone length ratios,
   axis conventions, typical values.
4. **Noise and framing model.** First session comparison (2026-11-01): MediaPipe's landmarks differ
   systematically from MHR anatomy: eye points 3.4 cm apart (MHR 6.3), hip points 6 cm wider,
   arms ~17% and torso ~12% shorter. Needs a per-landmark bias layer, randomized around the
   calibrated values, plus input normalization (e.g. by shoulder width).
   Done 2026-11-01 (`mttrain.mpstyle`, `mttrain.noise`, `mttrain.calibrate`): placement biases (both
   spaces) vs world-only distortions, calibrated against the fitted CMU population; AR(1) jitter,
   visibility and drifting guesses by distance outside the frame, lagged visibility, glitches,
   dropouts; all randomized per sequence. Synthetic vs real: proportions and visibility match,
   rest jitter ~0.6-1x real. Limits: one session, one person; how wrong MediaPipe's far
   out-of-frame guesses are is unknown (seated-v2 sessions needed). Calibrate on the recorded sessions: jitter spectrum per landmark
   at rest, depth noise vs 2D noise, visibility values, dropouts. Out-of-frame joints: imitate
   what MediaPipe does with arms below the frame (from the arms-below-frame segment; more
   seated-v2 sessions improve this). This is the hardest part and drives success criterion 1.
5. **Seated motion.** (done 2026-11-01, `mttrain.seated`: seated legs, damped pelvis lean/turn,
   breathing and sway, per-arm rest postures lap/desk/hanging from IK templates, limits clamped;
   camera framing sampled around the session's; seated hips get lower visibility as measured.) CMU upper body (spine and up) on a seated lower body (fixed hips/legs in a
   chair pose), plus idle sway/breathing.
6. **Model + training.** (code done 2026-11-02: `mttrain.skeleton/features/model/losses/dataset/train`;
   17 rotations, 1.14M parameters, 0.7 ms/frame on one CPU thread; first run `m1-first` training.
   Joint limits in training use a per-joint rotation-vector box measured on the clamped targets,
   a cheap proxy for MHR's Euler limits.) Inputs: landmarks + visibility at time offsets, differences between
   them. Output: 6D rotations for the body joints. Losses as in decision 2; joint-limit penalty
   from MHR's `[Limits]`. Train on the 3090. Track held-out synthetic metrics.
   First run (`m1-first`, 40k steps, ~78 min on the 3090): held-out subjects, synthetic: wrist
   position 4.3 cm (seated 4.1), upper arm 11 deg, head 1.1 cm; wrist rotation stays ~23 deg (hand
   orientation barely observable from pose landmarks; M4 hand head). On the recorded session vs
   baseline: shoulders and hands much calmer (hand rest noise with arms below frame 2.5 vs 9.9 deg,
   shoulder twitches 1.9 vs 13/min, no added lag), but arms noisier (rest noise 5.0 vs 3.7 deg) and
   the head never moved. Cause: the CMU fits constrained only positions, so head orientation and
   hand roll (hence forearm twist) were unconstrained in the targets. Fixed in the fit (head
   orientation constraint from CMU's head bone, thumb pairs); refit + retrain `m1-v2`.
   Second run (`m1-v2`, refit data, same settings): held-out synthetic wrist 4.3 cm (seated 4.2),
   upper arm 11.3 deg, forearm 6.9 deg, head direction ~7 deg (neck/head split in the targets is
   arbitrary; local head-group error ~12 deg). Recorded session, raw output vs baseline /
   baseline-rendered: rest noise shoulders 0.7 vs 3.4 / 2.6, arms 3.1 vs 3.7 / 3.2, hands 1.3 vs
   7.1 / 6.0, torso+head 1.0 vs 0.4 / 0.2 deg; twitches/min shoulders 0, arms 9.4 vs 16.9 / 45,
   hands 0 vs 15 / 54; lag 0 ms; 0.26 ms/frame. Follow-ups: consistent neck/head split in the fit,
   torso jitter, render filter for model output, hand-position and exact limit metrics in the
   harness, seated-v2 sessions for the go/no-go.
7. **Inference on sessions.** (`mttrain.infer` writes harness pose tracks. Render stage done
   2026-11-07: baseline CLI `--poses` filter-only mode, harness `--system m1-rendered=<poses>`;
   filtering the baseline's own solver output this way reproduces `baseline-rendered` exactly.
   m1-v2 through the filter, rest noise vs baseline-rendered: shoulders 0.5 vs 2.6, arms 2.7 vs 3.2,
   hands 0.9 vs 6.0, torso+head 0.5 vs 0.2 deg; lag 75-100 ms like the baseline's.)
   Criteria framing changed (owner, 2026-11-08): the model replaces the filter too, so its output is
   upsampled to 60 Hz without smoothing (`mteval.upsample.glide`, harness `--system m1-glide=<poses>`)
   and compared against `baseline-rendered`. m1-v2 glide vs baseline-rendered: rest noise shoulders
   0.6 vs 2.6, arms 3.0 vs 3.2, hands 1.2 vs 6.0, torso+head 0.9 vs 0.2 deg; arm twitches/min 13.1
   vs 45 (3.8 and 22.5 in the two still segments); lag 90 vs 70-95 ms (a causal 60 Hz display of
   10 FPS input costs at least ~50 ms on the harness's midpoint-stamped metric). Run the model on `sessions/*/landmarks.jsonl`, convert MHR → VRM,
   write a pose track the harness accepts, apply the same render stage as `baseline-rendered`,
   and produce the harness report against the baseline.
7b. **Real-vs-synthetic gaps and fixes (2026-11-08, `m1-v4`).** New harness metrics: hand position
   on the performer's skeleton (criterion 7), `global:head` (head orientation as the product of the
   torso/neck/head bones; the baseline outputs fewer torso bones, so per-bone torso numbers weren't
   comparable), and `mttrain.limitcheck` (criterion 8: IK of MHR parameters to the predicted
   orientations, checked against [Limits], on the session and stressed copies). Found and fixed:
   - MediaPipe's pose2d z is ~3x geometric depth; synthetic image z was unscaled. m1-v2 misread
     resting arms by 35-40 degrees on real input (forearm vs image). `mpstyle.IMAGE_Z`.
   - Synthetic still jitter was ~0.5x real (world); recalibrated per group, plus image-z jitter.
   - Camera height range raised (real torso lean toward the camera 6-28 degrees).
   - CMU head calibration offsets (per subject, up to 180 degrees) were copied into the targets;
     head joint at its limits in 23-77% of frames. Removed per subject, neck/head prior; 1-4%.
   - Rest postures faded out at crop ends, where the targets are: targets rarely saw a full rest
     pose (both generators fixed).
   - Training data now comes from a batched GPU generator (`mttrain.gpugen`, RTX 4060): 36 min per
     40k-step run on about one CPU core (was ~90 min with 10 CPU workers; the CPU path plus an
     unbounded refit crashed the machine). Heavy CPU jobs run under `tools/guarded-run.sh`.
   m1-v4 on the session (glide vs baseline-rendered unless noted): forearm direction vs image 2-15
   degrees (m1-v2 33-42, baseline 4-14); hand position in view 5.8 vs 4.7 cm (raw 5.2 vs 3.9; m1-v2
   13.9); rest noise head_global 0.8 vs 1.1, shoulders 0.6 vs 2.6, arms 3.5 vs 3.2, hands 1.6 vs 6.0
   deg; arm twitches 26 vs 45/min; lag 90 vs 95 ms; MHR limit violations >1 deg in 1.3% of frames
   (m1-v2 44%), none >6 deg; arms out of frame 5.4% / 0.7%. Earlier "calm arms" partly came from
   the model ignoring its input. Open: arm rest noise (now tracks MediaPipe's noise), criterion 7
   (the baseline solves from the very landmarks the metric compares against; m1 outputs anatomical
   rotations, so beating it by 30% may not be a fair target).
7c. **Coherent noise, criterion 7, rendered sessions (2026-11-09).**
   - Real MediaPipe rest noise is coherent within an arm (shoulder-elbow 0.32, elbow-wrist 0.52,
     wrist-hand 0.99; synthetic was 0): `noise.MIX`. `m1-v5` trained with it. On the real session vs
     m1-v4 (glide): forearm direction 6.6 vs 8.9 deg, arm twitches 22.5 vs 26.2/min, arm rest noise
     unchanged (3.5 deg), hands slightly noisier (2.4 vs 1.6 deg).
   - Criterion 7 restated (PLAN.md §6, owner): 2D wrist error vs image landmarks (arm lengths from the
     image; MediaPipe's world landmarks shorten arms ~30%), forearm direction guard, contact segments
     (palms together, hand to face) added to seated-v2. m1-v4 vs baseline-rendered: 2D 5.2 vs 7.2 cm,
     forearm direction 8.9 vs 8.8 deg.
   - Rendered sessions (`mttrain.render`, Blender EEVEE, MHR mesh, held-out CMU motion, replayed through
     the recorder headless; landmarks verified to match the training projection and left/right
     convention). Motion set (8 x 20 s, very active clips) and rest set (6 x 15 s, held posture,
     arms resting). Arm direction errors vs truth (median; MediaPipe raw / baseline / m1-v4 / m1-v5):
     motion: forearm 37.6 / 37.4 / 32.5 / 34.1 deg, wrist 22.9 / 23.8 / 21.1 / 21.5 cm (upper arm in the
     image plane only ~7 deg: the error is depth); rest: upper arm 32 deg for all, forearm 39 / 39 /
     35 / 38, wrist 29 / 29 / 28 / 30 cm.
   - Finding: MediaPipe's world landmarks flatten depth. Per axis vs truth (shoulder-width units):
     image plane slopes 0.9-1.1 (r 0.85-0.93); toward the camera 0.42 (upper arm, r 0.70) and 0.32
     (forearm, r 0.33); lengths kept, so the in-plane part is stretched. Synthesis assumes MediaPipe's
     world depth is right (plus noise): the models inherit the flattening. Caveats: plain mannequin
     renders (no clothes, hair, texture), 3 held-out body shapes; real people may differ.
   - Also found: seated idle motion (AR(1) signals) is rough frame to frame (arm parameters ~7 deg/s
     each, 16-20 deg/s per arm), so the rest set's ground truth is never still and training targets
     carry the roughness; real idle motion is smooth. Not fixed yet.
7d. **Depth flattening in the synthesis (2026-11-09).** Owner: train the model to undo the flattening
   (vspark compensates with `ikCalibration.zScale = 3` on world-landmark depth, IK mode only).
   - Calibration (renders, MediaPipe vs synthesis of the true pose, per segment, camera depth):
     MediaPipe = gain x true + bias, segment length kept. Gains: upper arm 0.37, forearm 0.33, hand
     0.18, shoulders 0.60, hips 0.47, torso 0.42 (bias -10 cm: leans toward the camera), head 0.25
     (-11 cm), head yaw 0.47. Image z is ~3x world z on every segment, so the old synthesis (true world
     depth, image z 3x geometric) was ~3x too deep in both spaces. Extra depth error during motion
     4-7 cm (~0.5 s), under 1.5 cm at rest: not modelled.
   - `m1-v6` (flattening at the render strength): renders much better than m1-v5 (motion set, arms in
     view: forearm 29.3 -> 22.8 deg, wrist 18.0 -> 14.4 cm; rest set wrist 35.6 -> 24.7 cm), but the
     real session worse (2D wrist 6.0 -> 9.1 cm). On the real session MediaPipe's forearm depth matches
     the image foreshortening (world and image in-plane fractions 0.68/0.68, 0.59/0.57); on renders the
     world forearm is stretched in-plane: real video looks less flattened than the plain renders.
   - FOV experiment (`render prepare --fovs`, 3 clips x 50/65/80/95 deg, constant framing, camera on the
     same line from the chest, 1.5-0.64 m): closer cameras flatten more (upper arm gain 0.24 -> 0.10,
     forearm 0.18 -> -0.03, torso 0.48 -> 0.36, neck 0.62 -> 0.44; left-right depth slightly less).
     Does not explain the real/render gap (the real session used the ~90 deg camera).
   - `m1-v7`: cameras FOV 50-130 deg (owner), distance 0.45-1.6 m, shoulder width 10-50% of the frame;
     gains change with camera distance (`DEPTH_SLOPE`) and a per-sequence relax factor (uniform 0-1)
     blends toward no flattening, so the model infers the strength. Real session (glide vs
     baseline-rendered): 2D wrist 6.7 vs 7.2 cm (m1-v5 6.0, m1-v6 9.1), forearm direction 7.1 vs 8.8 deg,
     arm rest noise 3.7 vs 3.2 deg (m1-v5 3.5), arm twitches 31.9 vs 45/min (m1-v5 22.5), lag 90 ms.
     Renders, arms in view, forearm deg / wrist cm (m1-v5 / m1-v6 / m1-v7): motion 29.3/18.0, 22.8/14.4,
     23.9/14.9; rest 44.7/35.6, 36.6/24.7, 27.9/19.9; FOV 50: 25.7/19.6, 23.8/19.7, 21.7/17.9; FOV 95:
     33.7/22.3, 30.2/22.4, 31.8/22.8. All systems get worse with closer cameras (MediaPipe forearm 30
     -> 41 deg).
   - Open: synthetic noise doesn't grow when the person is small in the frame (now down to 10% shoulder
     width); pinhole projection only (wide lenses add barrel distortion); real-video flattening strength
     needs ground truth (multi-view recordings).
7e. **Contacts (2026-11-09).** Owner: a contact render set, also the basis for follow-along recordings
   later; then contact-aware training.
   - `mttrain.choreo`: designed seated choreography (prayer palms, fingertip or palm on chin, cheek,
     forehead or mouth, near misses 8-14 cm), arm poses by MHR IK, min-jerk approach/hold/release; the
     holds are exact contact intervals. Rendered set `contact1` (8 x 20 s, held-out bodies). The
     contact metric (7b) on the true poses: palms 16.6 cm apart on MediaPipe's skeleton although the
     palms touch on the true body (the proportion problem in miniature).
   - Contact-aware training (`m1-v8`): a choreography pool on training bodies (1000 x 30 s; 100 on
     validation bodies) is 20% of the samples; contact loss `con` on the vectors between contact
     points (hands, nose) after FK on the true body, weighted exp(-(d/15 cm)^2). Synthetic validation:
     near pairs 3.5 cm (m1-v7 7.3).
   - `contact1`, 7b on MediaPipe's skeleton, |error| vs truth (m1-v7 / m1-v8): palms 3.9 / 4.9 cm
     (palms now too close on that skeleton), hand-to-face 11.1 / 7.9 cm (MediaPipe 7.7). On the true
     body (scratch `truebody.py`): palms 10.0 / 6.8 cm, hand-to-face 29.7 / 25.2 cm, near pairs
     (all frames) 23.4 / 14.2 cm.
   - **Hand-to-face is a depth problem from MediaPipe**: the error is almost all along the viewing
     axis (hand 30-37 cm toward the camera). During true contacts (hand within +-5 cm of the nose's
     depth) MediaPipe puts the hand 8-26 cm in front of the nose (world) and 22-100 cm (image z), the
     same as for near misses that really are 5-19 cm in front. The synthesis has no such bias, so the
     model takes the hand to be far in front, then undoes flattening on top. Unverified on real video:
     the recorded session has no contact segments (needs seated-v2).
   - Elsewhere m1-v8 is about level with m1-v7: motion renders arms in view 25.2 deg / 15.4 cm (m1-v7
     23.9 / 14.9), FOV 50 22.2 / 18.6, FOV 95 34.2 / 23.7; real session glide 2D wrist 6.4 cm
     (6.7), forearm 7.6 deg (7.1), arm rest noise 3.8 deg (3.7), arm twitches 30 / min (31.9). The
     rest set's arms-in-view figure is one arm in one sequence (seq02): 46.1 / 30.7 vs 27.9 / 19.9.
8. **Go/no-go write-up** in PLAN.md §6/§10 with the criteria table.

## Acceptance / verification

- Shared package tests pass; the spike's numbers are unchanged after the move.
- Synthetic landmarks look like real ones (side-by-side stats vs the recorded session).
- Model runs causally on a recorded session in < 2 ms per frame on CPU (ONNX or PyTorch).
- Harness report with the nine M1 criteria on at least three seated-v2 sessions (needs owner
  recordings; the existing seated-v1 session can be used for development).

## Open questions (ask before deciding)

- ~~How the `baseline-rendered` render stage is applied to model output~~: reuse the baseline's
  Node renderer with a pose-track input (owner, 2026-11-07).

## Output

Merge into `dev`.
