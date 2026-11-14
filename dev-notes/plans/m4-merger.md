# Plan: learned merger (body + hands + head/face)

> Branch: `feature/m4-merger` · Status: draft (open decisions below need the owner)

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

## Open decisions (owner)

1. **Head observation.** (a) Recommended: reduce the face mesh to a head pose (rotation, plus position and scale in
   the image) by fitting MediaPipe's canonical face model (Apache-2.0) to the 478 points, at runtime and on renders;
   in training, simulate that head pose from the true head with an error model calibrated on the renders (`head1` and
   the others), like the hand calibration. (b) Simulate the face points themselves from MHR's head mesh (needs a
   vertex correspondence to MediaPipe's topology) and let the merger read a subset. More faithful, much more work.
2. **Blendshapes in training.** Synthetic data can't pair expressions with motion. (a) Recommended: feed real
   recorded blendshape sequences (independent of the motion) with dropout, so the slot exists, the merger stays
   robust to them, and it learns from them once paired recordings arrive. (b) Keep the slot always empty until then.
   Either way blendshapes pass through to the output unchanged in this version.
3. **Time.** (a) Recommended: a short history of the part outputs at a few time offsets (like `m1`), so twist, rest
   and clipping decisions can use motion and how long confidence has been low. (b) Per frame only, temporal state
   left to the part models (`PLAN.md` §3.6 `[CLAUDE]`).
4. **Training data.** Recommended: a new merger pool of full MHR parameter sequences (resumable shards): `m1`'s body
   sources (seated clips, choreography, head moves) with procedural fingers, plus `hands.stage`/`hands.pairs`
   sequences with torso and head motion added. Observations and the frozen part models run on the GPU per batch.
   Clipping correction: part of the samples get observed hands pushed into the body (as MediaPipe does near the
   skin) while the target stays outside it.

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
