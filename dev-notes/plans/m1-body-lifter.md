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
2. **Batch CMU → MHR fit.** Fit the selected clips at 30 fps (`step=4` on 120 fps), store MHR
   model parameters + subject scales per clip in `data/m1/fits/`. Log residuals; drop clips with
   bad fits. Loosen the spine pairing (spike finding: spine scales lean on their limits).
   About 0.3 s per 100 frames, so the whole set takes minutes to an hour on CPU.
3. **Landmark synthesis.** Map MediaPipe's 33 pose landmarks to MHR joints/mesh points (face
   points like nose, eyes, ears, mouth corners from the mesh; hips/shoulders/elbows/wrists from
   joints with an offset). Produce both outputs the recorder stores: world landmarks (metres,
   hip-centred, MediaPipe axes) and image landmarks (`pose2d`, from a sampled camera: desk-webcam
   framing, FOV, distance, tilt). Validate against the real session: bone length ratios,
   axis conventions, typical values.
4. **Noise and framing model.** Calibrate on the recorded sessions: jitter spectrum per landmark
   at rest, depth noise vs 2D noise, visibility values, dropouts. Out-of-frame joints: imitate
   what MediaPipe does with arms below the frame (from the arms-below-frame segment; more
   seated-v2 sessions improve this). This is the hardest part and drives success criterion 1.
5. **Seated motion.** CMU upper body (spine and up) on a seated lower body (fixed hips/legs in a
   chair pose), plus idle sway/breathing.
6. **Model + training.** Inputs: landmarks + visibility at time offsets, differences between
   them. Output: 6D rotations for the body joints. Losses as in decision 2; joint-limit penalty
   from MHR's `[Limits]`. Train on the 3090. Track held-out synthetic metrics.
7. **Inference on sessions.** Run the model on `sessions/*/landmarks.jsonl`, convert MHR → VRM,
   write a pose track the harness accepts, apply the same render stage as `baseline-rendered`,
   and produce the harness report against the baseline.
8. **Go/no-go write-up** in PLAN.md §6/§10 with the criteria table.

## Acceptance / verification

- Shared package tests pass; the spike's numbers are unchanged after the move.
- Synthetic landmarks look like real ones (side-by-side stats vs the recorded session).
- Model runs causally on a recorded session in < 2 ms per frame on CPU (ONNX or PyTorch).
- Harness report with the nine M1 criteria on at least three seated-v2 sessions (needs owner
  recordings; the existing seated-v1 session can be used for development).

## Open questions (ask before deciding)

- How the `baseline-rendered` render stage is applied to model output (reuse the baseline's
  Node renderer with a rotation-track input, or reimplement).

## Output

Merge into `dev`.
