# Plan: legs, hip position and up direction in the body model and the merger

> Branch: `feature/legs-root` (from `feature/relational-retarget`, which builds on `feature/m4-merger`) · Status: in progress

## Goal

The avatar gets legs, a hip position and a gravity direction from the learned models, instead of standing rest
legs at a fixed height. Motivation (owner, 2026-11-14): hands resting on the thighs can't land on the avatar's thighs
while the avatar stands; the training data has always had full-body ground truth, and throwing the legs away was
never a decision (the body model's 17 joints came from the M1 plan's scope).

## Decisions (owner, 2026-11-14)

- Legs in **both** the body model and the merger. The body model reads hip/knee landmarks when they are in view;
  the merger refines with what it sees (hands resting on the thighs carry the lap's position).
- Desk use is the primary use case, but other movement (standing, moving around) must keep working. Sitting vs
  standing is **learned from the input**, not a context switch. If the lower body is too unstable at the desk, the
  consumer can mask it.
- **Hip position vector now** (pulled forward from M5, `PLAN.md` §3.9): **camera frame, metres, absolute per
  frame** (x right, y down, z forward from the camera; no deltas fed back), plus an **up direction** (gravity in
  the camera frame) as a separate output, so the floor frame is one rotation of the whole pose away and the tilt
  guess sits in one replaceable place (a calibration step can override it later).
- Rationale for the camera frame over the floor frame: every other output (the root orientation included) is
  already camera-relative; the camera's tilt and height are not observable from the body alone (tilt vs lean
  ambiguity); hip height above the floor follows from the leg pose with feet on the floor.

## Constraints

- Strictly causal, lightweight; the stateless history-window design stays (`mttrain.model.Lifter`,
  `mttrain.merger.model.Merger`).
- Resumable training (memory: runs must be resumable). `mttrain.train` is not yet: add `--resume`.
- Heavy jobs under `tools/guarded-run.sh`, one at a time.
- No regression on the arms: compare against m1-v11 / m1-v10 on validation and the rendered sets.

## Approach

1. **Outputs.** `skeleton.OUT_JOINTS` += `{l,r}_upleg`, `{l,r}_lowleg`, `{l,r}_subtalar` (VRM UpperLeg,
   LowerLeg, Foot; subtalar's parent in the output chain is lowleg, the intermediate foot joints keep their rest
   like the twist joints). Extra head on the Lifter: root position (x/z, y/z, log z, two log scales: image plane
   and depth) and up (3-vector, normalized).
2. **Targets.** Root position = camera-frame position of MHR's `root` joint (metres); up = camera rotation applied
   to MHR world +y. Both generators (numpy `TargetExtractor` for validation, torch `gpugen.geom.Targets` for
   training).
3. **Losses / metrics.** Laplace NLL on the root position (image plane and depth separately), angle of the up
   vector, leg joints in the rotation and FK position losses (knees, ankles). Metrics: leg degrees, root cm
   (lateral, depth), up degrees, seated vs standing hip height error.
4. **Train** m1-v12 (resumable), compare with m1-v11 on arms; evaluate legs/root/up on the rendered sets (ground
   truth from `seq.npz`), and sitting-vs-standing stability on the real seated sessions.
5. **Merger:** legs and root/up in its inputs (from the body model) and outputs (residuals), gap calibration for
   the new outputs, retrain (mg3), evaluate.
6. **Output path:** VRM legs through `LifterPose`; hip position and up in the pose files and avatar videos.
   Service/VMC output of the root comes after the owner has seen the videos.

## Out of scope

- Root translation smoothing/anchoring and calibration (rest of M5).
- Richer seated leg variety in the generator (crossed legs etc.): noted, later.
- Live service changes.

## Progress

- 2026-11-14: decisions recorded, branch created.
