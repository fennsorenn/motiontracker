# Plan: merger v2 — realistic part errors, pose deltas, responsiveness lever

> Branch: `feature/merger-deltas` (not started) · Status: draft, for the owner's review
> Planned 2026-11-23 with the owner. Decisions are tagged in `PLAN.md` §3.4 ("Merger v2").

## Goal

The live merger (`mg3`) follows the body model and loses what the hand model gets right. A merger that uses
whatever information is most reliable at the moment (above all the hand model and its markers when a hand is seen),
stays stable, and lets the user trade responsiveness against stability.

## Why (measured 2026-11-23, renders with ground truth: `data/m1/renders/contact1`, `data/m4/renders/hands1`)

- **Arms ~15° too low at every height** in the body model's output, uncorrected by the merger. Wrist height above
  the shoulder line (shoulder widths), output vs truth: hands down −1.77 vs −1.26; at the shoulders −0.27 vs +0.12;
  face/eye level −0.03 vs +0.44 (owner's "hands at eye level shown at shoulder height"). Not the MHR→VRM conversion
  (checked: it slightly raises hands up to head level) and not retargeting (off; never in the measurement).
- **The merger makes the hand worse than the hand-written combination**: hand-direction error 30.7° (combination)
  → 38.8° (merger); the hand model alone: 20.2°. Twist range (slope against the truth): combination 0.41 → merger
  0.23; the hand model alone 0.48. The conversion keeps twist (slope 0.98–1.00).
- **The hand model's image position matches the hand landmarks** (1.7 px at 320×240), but nothing downstream places
  the arm from it.
- **Pose landmarks are smoothed inside MediaPipe** (`LandmarksSmoothingCalculator`, One Euro): speed-, scale-, frame
  rate- and reset-dependent; the owner sees the pose markers trail the hand markers live. The pose wrist also sits
  8–15 px from the hand wrist (320×240).
- **Cause (`merger/batch.py`, `merger/gapcal.py`)**: in training, the body model's sim-to-real gap is per-joint
  random noise of the measured size (`degrade`), with no direction (no systematic arm bias), no lag (no landmark
  smoothing in the observation path), and no wrist offset. With those inputs the body model's arm is right on
  average, so the best strategy was to keep it; a merger with any output form learns the same from that data.

## Decisions (owner, 2026-11-23)

- Realistic body (and hand) errors in the merger's training data: must have.
- The merger outputs **deltas from its previous output** (the previous output is an input), so no input source is
  the starting point by design; which source to trust comes from the training data only.
- **Staged training**: on true previous poses first, then fine-tuned on its own output.
- A **responsiveness vs stability lever** as a model input, trained with lever-weighted losses, exposed in vspark.
- Rejected: a hand-led base pose (struggles with hand dropout: every loss of the hand switches the starting point);
  a geometric override after the merger (defeats the purpose of the merger).
- Proposed, not yet confirmed (`[CLAUDE]`): a middle training stage on deliberately wrong previous poses; an
  absolute-output control variant on the same new data, to tell what the data alone brings.

## Constraints

- Strictly causal, real-time; the merger stays small (CPU, one ONNX run per frame).
- Part models (`m1-v14` body, `p3` hands, face head pose) stay frozen, as for `mg3`.
- The runtime keeps working without the merger (`--no-merger`) and with the current `mg3` bundle until a v2
  bundle is shipped; v2 is a new bundle, not an in-place change.
- No CMU or CMU-derived data in git; weights only as release assets.
- Measurements on the renders as in this plan's "Why" are the reference: v2 must be compared on them, with the
  same scripts (see Acceptance).

## Files in scope (expected)

- `train/src/mttrain/merger/gapcal.py` — calibration extended to signed, pose-dependent bias, lag and wrist offset
  of the body model on real Holistic output, and hand dropout statistics.
- `train/src/mttrain/merger/batch.py` — the observation path with landmark smoothing (One Euro like MediaPipe's,
  calibrated), signed bias and lag in `degrade`, realistic hand dropouts; previous-pose inputs for the training
  stages; the lever value per sample.
- `train/src/mttrain/gpugen/` — only if the smoothing belongs in `GpuGenerator.observe` itself.
- `train/src/mttrain/merger/model.py` — previous output as input, delta output (composition on the previous
  global rotations), lever input.
- `train/src/mttrain/merger/train.py` — stages (true / perturbed / own-output rollouts over 1–2 s), lever-weighted
  losses (lag/position vs acceleration).
- `train/src/mttrain/merger/export.py`, `core/src/mtcore/merged.py` — runtime state (previous output), reset on
  tracking loss and `reset()`, lever input; bundle format version.
- `service/src/mtservice/cli.py` — lever option and stdin command; vspark: a slider in the Learned Tracking
  properties.

## Out of scope

- The hand model's own twist range (about half the true range even alone): a separate task on `p3`'s training.
- Retraining the body model (its arm bias stays; the merger learns to correct it).
- GPU detection, frame rate defaults, history shift (separate open items, `PLAN.md` §7.7).

## Approach and estimates

1. **Realistic part errors in the training data — 1–1.5 days.**
   - Measure on the renders: the body model's signed per-joint error as a function of arm elevation and hand height
     (the arm-down bias), its lag against the truth, the pose wrist's offset from the hand wrist, MediaPipe's pose
     smoothing (fit One Euro parameters from render landmarks against the projected truth, or from pose vs hand
     wrist), hand dropout runs (~3 h).
   - Build into batch generation: smoothing of the simulated pose landmarks before the frozen body model (so its
     outputs lag and bias the way they do on real input), the measured wrist offset, signed bias where the
     simulation still falls short, realistic hand dropouts (~5 h).
   - Check: the body model's errors on the new synthetic batches match its errors on the renders (bias by height,
     lag, size), as `gapcal` does today for the size (~2 h).
2. **Delta output — 0.5–1 day.** The merger gets its previous output (globals, 6D) as an input and outputs a
   correction applied to it; part outputs and their history stay inputs. Runtime: keep the previous output; on
   start, reset and after tracking loss, start from the hand-written combination (`combine`).
3. **Staged training — 1.5–2 days.**
   - Stage A: previous pose = the truth (~1 h GPU).
   - Stage B `[CLAUDE]`: previous pose = the truth with deliberate errors (offsets, slow drift, the body model's
     typical bias), so it learns to pull a wrong pose back (~1 h GPU).
   - Stage C: own output as the previous pose over 1–2 s stretches (rollouts with gradients through the stretch),
     with dropouts and sudden moves in the data (code ~5 h, training 2–4 h).
4. **Responsiveness lever — ~0.5 day, inside stage 3's runs.** A scalar input per sequence, sampled uniformly in
   training; it weights the position/lag losses up and the acceleration loss down (or the reverse). Check that
   outputs move monotonically with the lever (lag and shake per lever value on the renders).
5. **Comparison — ~1 day.** On the renders, against `mg3`, the hand-written combination and the hand model alone:
   the measurements of "Why" (wrist height by true height, hand-direction error, twist slope, lag), plus shake
   (frame-to-frame and after vspark's Minimal filter) and recovery after hand dropouts. Control variant
   `[CLAUDE]`: absolute output trained on the same new data (code ~2 h, ~1 h GPU).
6. **Live — ~0.5 day.** Runtime support, vspark slider, release; the owner compares against `mg3` live.

Total about 5–7 working days. Risks: stage C unstable (rollouts drift or diverge) and the synthetic errors not
matching the renders on the first try, each +1–2 days.

## Acceptance / verification

- The scripts of 2026-11-23 rerun on the v2 bundle: wrist height within ~0.1 shoulder widths of the truth in every
  height bin (today up to 0.47 off); hand-direction error at or below the hand model alone's 20°; twist slope at
  least the hand model's 0.48; wrist lag at or below the body model's 50 ms (`mg3`: 70 ms).
- Shake at the most stable lever setting no worse than `mg3`; at the most responsive, lag clearly below `mg3`.
- No drift over long sequences (error at the end of 20 s sequences not above the start); clean recovery after
  hand dropouts.
- Owner's live check in vspark: hands at eye level shown at eye level; pose markers' trailing no longer visible
  in the avatar's arm.

## Output

Branch `feature/merger-deltas`; merge into `dev` when the comparison is done and the owner has compared live.
