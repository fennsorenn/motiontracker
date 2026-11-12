# Plan: learned hand model (fingers from MediaPipe hand landmarks)

> Branch: `feature/m4-hand-model` (from `feature/m2-face-hands`) · Status: draft (2026-11-12)
> This plan is the seed context for a worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

A small learned, strictly causal model that turns one hand's MediaPipe landmarks into stable finger
rotations, replacing the ported algorithmic finger solve (`service/src/mtservice/hands.py`) in the live
runtime. Part of M4 (`PLAN.md` §11); the first part head after the body lifter.

## Decisions (2026-11-12)

- Training motion: **procedural finger poses** within MHR's finger limits, plus a gesture vocabulary,
  joined by smooth transitions (`PLAN.md` §5.6). `[USER]` Known hand mocap datasets are non-commercial.
- Training inputs: **synthetic MediaPipe-style hand landmarks** from MHR forward kinematics plus a noise
  and bias model, calibrated on rendered hands run through real MediaPipe (as M1 did for the body). `[USER]`
- Model input: **the hand's own landmarks only** for the first version. `[USER]` This defers the
  standing decision that part heads get full-pose context (`PLAN.md` §3, `[USER]`); the input layout
  leaves room to add body context later.
- Evaluation: **rendered hands with exact ground truth** (Blender, real MediaPipe on the renders),
  learned model vs the ported solve. `[USER]`
- Output: fingers only (MHR's 27 finger parameters per hand → VRM finger bones via `mtcore.convert`).
  Wrist and forearm twist stay with the body model until the merger. `[CLAUDE]`
- One model for both hands, the left hand mirrored into the right hand's frame. `[CLAUDE]`
- Synthesis on the GPU like `gpugen`; model and training loop patterned on M1's lifter. `[CLAUDE]`

## Facts (checked 2026-11-12)

- MHR: per hand 27 finger parameters (thumb 7, index/middle/ring/pinky 5 each), all with `[Limits]`
  entries; joints `l_thumb0..3`, `l_index1..3`, `l_middle1..3`, `l_ring1..3`, `l_pinky0..3`, `*_null`
  fingertip ends, `l_wrist`, `l_wrist_twist`. CMU has no finger data, so M1's fits keep fingers at rest.
- `mtcore.convert` already maps MHR fingers to VRM finger bones (direction matching, M3 spike).
- MediaPipe Holistic finds hands from the body pose (hand crops come from the pose's wrist), so hand
  renders must show a body, not a floating hand.
- The ported solve reacts strongly to tiny landmark changes (0.58 deg from 10 µm rounding), which is
  where the learned model should gain.

## Constraints

- Strictly causal, lightweight: target well under 1 ms/frame for both hands on CPU (body lifter ~0.3 ms).
- Permissive only: no non-commercial hand data or models (`docs/LICENSES-REGISTER.md`).
- Heavy jobs through `tools/guarded-run.sh -c 8-15`, one at a time; GPU generation as in M1.
- Renders that use CMU motion are CMU-derived: gitignored `data/`, never committed. Purely procedural
  hand renders are not, but keep them in `data/` too.

## Files in scope

- `train/src/mttrain/hands/` (new): procedural finger motion, hand landmark synthesis (MHR joints →
  MediaPipe's 21 points), noise model, dataset/generator, model, training, inference, export.
- `train/src/mttrain/render.py`, `tools/render/`: renders with finger motion (body + hands in view).
- `service/src/mtservice/`: hand model runner next to `hands.py`, selectable by flag.
- `docs/LICENSES-REGISTER.md`, `dev-notes/`, `PLAN.md`.

## Out of scope

- Body context as input (later), wrist/forearm twist, merger, two-hand interaction, contacts.
- Label-free metrics on real sessions, a new recorded session, comparison videos (not chosen for now).

## Approach

1. **Hand skeleton spike.** MHR hand joints vs MediaPipe's 21 landmarks (which MHR joint sits where,
   thumb especially: `thumb0..3`); finger parameter axes and limits; MHR → VRM fingers round trip with
   `mtcore.convert`; mirroring left ↔ right.
2. **Procedural finger motion.** Random poses within limits with finger coupling (neighbouring fingers
   curl together, DIP follows PIP), a gesture vocabulary (open, relaxed curl, fist, point, pinch, OK,
   counting, grip, splay), minimum-jerk transitions, holds and small idle motion. Check by rendering a
   few sequences.
3. **Rendered hands with ground truth.** Extend the render pipeline: seated body with hands in view and
   procedural fingers, through the recorder's MediaPipe build. This gives the evaluation set and the
   calibration data for step 4.
4. **Hand landmark synthesis and noise.** FK → MediaPipe world (metric, hand-centred) and image
   landmarks, with biases and depth flattening measured on the renders (as `mpstyle` for the body);
   jitter, dropouts, wrong-hand and lost-tracking cases. Compare synthetic vs rendered statistics.
5. **Model and training.** Per-hand causal temporal model over a short history of landmark frames;
   outputs 27 finger parameters (or finger joint rotations) and confidence; losses on angles and
   fingertip positions; joint limits.
6. **Evaluation on the renders:** finger angle error and jitter, learned vs ported solve, per finger.
7. **Runtime.** Export (ONNX), runner in `service/`, flag to choose ported or learned; timing.

## Acceptance / verification

- On the rendered evaluation set, the learned model beats the ported solve on finger angle error and
  jitter (thresholds to agree with the owner once the first numbers exist).
- Runtime cost for both hands reported; runtime tests pass.

## Output

Commits on `feature/m4-hand-model`; report to the owner after each step's numbers.

## Progress (2026-11-12)

- Step 1 (hand skeleton spike) done:
  - MediaPipe's 21 points ↔ MHR joints: 0 `wrist`; thumb 1-4 = `thumb1`, `thumb2`, `thumb3`, `thumb_null`
    (MHR's `thumb0` is an extra base joint, 1.8 cm from the wrist); fingers 5-20 = `<f>1`, `<f>2`, `<f>3`,
    `<f>_null` for index, middle, ring, pinky (MHR's `pinky0` is an extra metacarpal joint). Distances
    from the wrist agree (thumb CMC 4.1 vs 3.8 cm on the session, index MCP 8.6 vs 9.6, pinky 7.5 vs 8.1).
  - MediaPipe bias to model in step 4: its proximal phalanx is about 1 cm shorter than MHR's on every
    finger (index 2.6 vs 3.8 cm) and the metacarpal about 1 cm longer, i.e. its knuckle point sits past
    the joint centre.
  - Parameters: `<f>{1,2,3}_rz` curl toward the palm (positive = flex); `<f>1_ry` spreads; `<f>1_rx`
    (axial roll, limits ±0.3-0.5) does not move the fingertip at all: unobservable from landmarks, keep
    at 0 and don't predict it. Thumb: `thumb0_ry/rz` (large base motion), `thumb1_ry/rz` (`thumb1_rx`
    moves the tip 0.4 cm), `thumb2_rz`, `thumb3_rz`. About 22 observable parameters per hand.
  - Right hand = left hand mirrored across x for the same parameter values (max 0.001 cm): one model for
    both hands, the left mirrored.
  - `mtcore.convert` already maps MHR fingers to VRM finger bones; nothing new needed there.
- Step 2 (procedural finger motion) done: `mttrain.hands.skeleton`, `mttrain.hands.motion` (20db6c1). 16
  gestures (thumb poses by IK, pulled toward a natural thumb), coupled random poses (random thumbs are
  blends of the gestures'), holds and minimum-jerk transitions, per-finger timing offsets, drift,
  tremor, taps. Previews sent to the owner (gesture sheet, 12 s video).
- Step 3 (rendered hands) done: `render prepare --hands` with `mttrain.hands.stage` (hands-in-use arms by
  IK in front of chest and face, palms any way), procedural fingers, 60-95 deg cameras; `fingers.npz` and
  finger bones in `gt.jsonl` (8e6c34b). MediaPipe finds the hands in about half the frames, like on the
  real session. Known flaw: an arm can pass through the torso (IK has no body collision).
- Evaluation (`mttrain.hands.evaluate`, bc489b1): segment directions in a palm frame built from the hand's
  own points, system vs MHR truth. First numbers (2 test sequences): ported solve on MediaPipe median
  25.6 deg (p90 58.6), on perfect landmarks 12.2 deg (its own model's floor).
- **Conversion bugs found and fixed** (749b6dc, `mtcore.convert`): the avatar's palm sat 25.3 deg rolled
  against MHR's (hand D was the minimal rotation of the finger direction only), and distal finger bones
  had D = identity (~45 deg off). Ground truth replayed on the avatar is now exact (test). Live runtime:
  the body model's hand bones turn by 25.3 deg, nothing else changes. Regenerated: m1-v10's offline
  session poses (runtime test reference) and the ground truth of every render set (`render gt`).
- Step 4 (synthetic observations) done: `mttrain.hands.calibrate` on `hands1` (10 x 20 s rendered): MediaPipe
  pulls hands toward an average shape (per-coordinate slope 0.52; 0.45 on `hands2`), residual 0.13 palm
  lengths, lag-1 correlation 0.85 at 10 Hz, mirrored (palm/back confused) 10%, 77% of fully visible hands
  detected. `observe.Observer` reproduces this with per-sequence randomization. Shared input form:
  `mtcore.hand_features.canonical` (MediaPipe's hand world landmarks are not camera-aligned).
- Step 5-6 (model, training, evaluation): `HandNet` (8 frames at 10 Hz -> 23 finger parameters), GPU
  training (`mttrain.hands.train`, 20k steps in 11 min). `h1` on the evaluation set `hands2` (8 x 20 s,
  seeds not used for anything else), segment direction error median / p90, motion vs truth:
  constant average hand 31.0 / 86; ported solve 28.4 / 68, x5.0; **h1 13.2 / 47, x2.5**. On `hands1`
  (calibration set): ported 26.0, h1 11.0. Caveat: rendered finger motion comes from the training
  generator. `h2` adds a velocity loss against the jitter.
- `h2` (velocity loss, `--w-vel 2`) on `hands2`: 13.2 / 47 deg, motion x2.2 (h1 x2.5): same accuracy, a
  little less jitter.

## Owner feedback on h2 and v2 decisions (2026-11-12)

Comparison video (ported vs h2 on the real session): "surprisingly good". Weaknesses: fingers other than
the index are weak (counting on one hand doesn't show), wrist bends not captured, fingers sometimes
slightly off their natural bend axis. Decisions for v2: `PLAN.md` §5.6 (absolute hand rotation and
position output, position/rotation history input, hands as a pair, interactions/flips/swaps explicitly in
the training data). Order: wrist first, then the finger fixes.

Next steps (v2):
1. Measure MediaPipe's flips and swaps on the existing renders (durations, when they happen).
2. Two-hand interaction choreography (IK: palms together, fingertip to other hand, hand over hand,
   crossing, clapping; real fast palm turns), rendered and run through MediaPipe; calibrate the image-space
   hand landmarks (orientation, position) and interaction failures (swaps, merges, losses).
3. Pair model with absolute rotation and position; training data with interactions, flips and swaps.
4. Wrist evaluation on renders vs the body model's hand bone; combination step; comparison video.
5. Finger fixes.

## Recordings and detectors (2026-11-12)

- `hands-v1` recorder protocol (128 s). First take `sessions/2026-11-12T21-40-58-session`: framed on torso
  and hands, no head: Holistic found the face in 0% and hands in 0-19% of frames (it finds hands from the
  body). Protocol now asks for head and shoulders in view and has no desk steps. Retake
  `sessions/2026-11-12T21-56-29-session`: face and pose 100%, hands 70-100% per segment.
- Hand Landmarker (standalone) evaluated: see `PLAN.md` §5.6. The retake is the real video where both
  detectors can be compared directly.
