# Plan: learned merger (body + hands + head/face)

> Branch: `feature/m4-merger` · Status: in progress

## Goal

One learned step that joins the part models into a single consistent pose: the body model (`m1-v10`), the pair hand
model (`p3`) and the head/face (Holistic face mesh, MediaPipe blendshapes). It replaces the hand-written combination
(`mttrain.hands.wrist`) and fixes what no part model can see alone: forearm twist and wrist, the body's hand when the
hand model is unsure or the hand is out of view, hands and arms clipping into the body, and the head's nod/tilt/turn
against the chest (which `m1` under-reads, head-vs-chest slope ~0.2 against a target of 0.6).

## Decisions (owner, 2026-11-14; `PLAN.md` §3.4)

- Scope: body + hands + head/face, built and trained together ("all at once"), not staged.
- Inputs, first version: the part models' outputs only (rotations, confidences, the hand model's image position and
  size). Internal features follow later, with feature dropout.
- Output: residual corrections on top of the combined part outputs (`wrist.combine` as the starting point), so the
  merger switched off still gives a sensible pose and parts stay swappable.
- Hand-body clipping correction data belongs here (`PLAN.md` §5.6, 2026-11-13).
- Part models stay frozen while the merger trains (`PLAN.md` §3.4 training order).

## Facts (checked 2026-11-14)

- `m1` (`mttrain.model.Lifter`): per joint of `OUT_JOINTS` (root, spine0-3, neck, head, clavicles, upper/lower arm,
  wrist twist, wrist per side) a rotation and a log-scale (confidence). Torch generator `gpugen` builds its inputs on the
  GPU from MHR parameter sequences (CMU seated clips, seated synthesis, contact choreography, head moves).
- `p3` (`mttrain.hands.model2.PairNet`): per hand 27 finger parameters + log-scales, camera-frame hand rotation +
  log-scale, image position, log size; inputs from `observe2.PairObserver` (torch) with a detector score.
- Hand pools (`data/m4/pool3`) store derived arrays only (hand joints, body landmarks, finger parameters), not the
  full MHR parameter sequences the merger needs as ground truth. The generators that make them (`hands.stage`,
  `hands.pairs`) do produce full sequences (`mp`), with a mostly still torso and head.
- Rendered sets with exact ground truth and real Holistic output, face mesh included (478 points): `data/m1/renders`
  (`v1`, `rest1`, `contact1`, `head1`, `fov1`) and `data/m4/renders` (`hands1-3`, `pairs1-3`; also Hand Landmarker).
- No synthetic face mesh exists yet. No paired data of expressions and body motion exists (needs the ARKit + webcam
  recordings, `PLAN.md` §3.11).

## Decisions on the design (owner, 2026-11-14)

1. **Head observation: head pose from the face mesh.** Fit MediaPipe's canonical face model (Apache-2.0) to the 478
   points (rotation, plus position and scale in the image), at runtime and on renders; in training, simulate that head
   pose from the true head with an error model calibrated on the rendered sets. (Rejected for now: simulating the face
   points from MHR's head mesh.)
2. **Blendshapes in training: real, unpaired.** Real recorded blendshape sequences, independent of the motion, with
   dropout. Blendshapes pass through to the output unchanged in this version.
3. **Time: short history** of the part outputs at a few time offsets (like `m1`).
4. **Training data: a new combined pool** of full MHR parameter sequences (resumable shards): `m1`'s body sources
   with procedural fingers, plus `hands.stage`/`hands.pairs` sequences with torso and head motion added.
   Observations and the frozen part models run on the GPU per batch. Clipping correction: part of the samples get
   observed hands pushed into the body while the target stays outside it.

5. **Smoothing through training (owner, 2026-11-14): jitter on the input, none on the output.**
   - Input: on top of the jitter the frozen part models already pass on from noisy observations, extra
     high-frequency jitter on the merger's inputs (part rotations, hand position/size, head pose), its strength
     calibrated on the part models' frame-to-frame noise on real sessions (rest segments) so it is realistic, not
     larger.
   - Output: targets free of jitter. The current generators put some into the truth: `m1`'s idle life is AR(1)
     noise (rough frame to frame), and the hand motion adds an 8-12 Hz finger tremor (`hands.motion._idle`). Merger
     targets use smooth idle signals (low-pass) and no tremor; deliberate fast motion (taps, sharp starts and stops)
     stays, so the merger learns to filter noise, not to lag (`PLAN.md` §3.6: hedging). Acceleration/jerk loss on
     the output against the truth's.
   - Check: lag and peak-velocity metrics on the rendered sets alongside rest jitter, so smoothing doesn't buy lag.

## Approach

1. Head observation: the canonical-face fit (decision 1) as a shared function (runtime and analysis); measure it
   against the truth on all rendered sets; error model and its calibration.
2. Merger pool: generator, shards, resume, mesh checks reused from `hands.collide`.
3. Batch builder on the GPU: FK → body landmarks (`gpugen`) → `m1`; hand observations (`observe2`) → `p3`; head
   observation; blendshape slot; the combination (`wrist.combine`, differentiable) as the base pose.
4. Merger model and losses: residuals per bone, rotation and FK position losses, confidence (Laplace) outputs,
   body penetration penalty, relations between hands and face/body.
5. Training (`--resume`, checkpoints with optimizer/scheduler/RNG state) under `tools/guarded-run.sh`.
6. Evaluation on all rendered sets against `m1` + `wrist.combine` and real-session videos (seated-v1, hands-v1).

## Acceptance / verification

- Rendered sets: per-bone global rotation error and hand position relative to the body no worse than the part models
  plus `wrist.combine` anywhere, better at wrists, out-of-view hands and contacts; body penetration depth lower;
  head-vs-chest slope on `head1` closer to 1.
- Real sessions: avatar videos with face (seated-v1, hands-v1) judged by the owner.

## Out of scope

- Runtime integration (`service`): a follow-up plan once a merger checkpoint is accepted.
- Internal features as inputs; joint fine-tuning of parts and merger; the learned retargeting layer (§3.10a); gaze;
  a learned face model; feeding the merged pose back to the part models (§3.5).

## Output

Merge into `dev` when the owner accepts the evaluation.

## Progress (2026-11-14)

- Step 1 done: `mtcore.facepose` (canonical face fit, metadata read from the Face Landmarker bundle without
  protobuf) and `mttrain.merger.facecal` (calibration on 9 rendered sets, `data/merger/calib/face.json`). Face-mesh
  head pose against the truth: median 6.8 deg over all sets (p90 14), slopes turn 1.10 / nod 0.93 / tilt 0.90, face
  seen in 81% of render frames; error mostly a per-sequence offset (nod 6.3 deg between sequences, 2.5 within) plus
  ~1 deg jitter per 10 Hz sample. `m1-v10`'s head on the same renders: 14-19 deg.
- Step 2 running: merger pool `data/merger/pool` (`hands.pool2 --full --no-tremor`, 4 x 380 train draws + 110 val,
  resumable).
- Step 3 done: `mttrain.merger.batch` (`GpuGenerator` split into `crop`/`observe`, body model batches verified
  bit-identical; shared camera for body and hands; frozen `m1-v10` and `p3`; simulated face head pose as slow drift
  plus white jitter; real blendshapes from `data/videos/*-full/face.landmarks.jsonl`, 30% dropout; base pose =
  hand-written combination in the body model's frame; clipping correction: 30% of samples with observed hands near
  the torso/thighs pushed 1-4 cm into the body). On a small test pool the base pose already halves the body model's
  head error (12.4 -> 7.5 deg) and cuts wrists 30 -> 23-25 deg.
- Input jitter calibration: synthetic part outputs already jitter about as much as real ones at rest (body head 0.7
  vs 0.6-0.9 deg, body hands 2.4 vs 1.7-2.9 deg per 10 Hz frame; hand model 3.9 synthetic vs ~2 real), so the extra
  input jitter is small (default 0.75 deg).
- Step 4 done: `mttrain.merger.model` (residuals over offsets 0, 1, 2, 4, 8 frames) and `mttrain.merger.train`
  (resumable; validation compares merger, base pose and body model).
