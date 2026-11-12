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
