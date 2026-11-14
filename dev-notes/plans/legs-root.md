# Plan: legs, hip position and up direction in the body model and the merger

> Branch: `feature/legs-root` (from `feature/relational-retarget`, which builds on `feature/m4-merger`) · Status: trained and evaluated (m1-v14, mg3), awaiting the owner's review

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
- 2026-11-14: steps 1-5 done, step 6 partly (pose files carry leg bones and root/up; videos show legs, root/up not
  applied). Status: trained and evaluated, awaiting the owner's review.
  - Weights lesson: confidence-weighted leg/root gradients swamped the arms under the global grad clip 1.0
    (m1-v13 step 5000: grad norm legs 91, root 120, upper 8; arms ~5 deg behind m1-v11 in m1-v12/v13). Fix: leg and
    root weights 0.05 (`train.py` `WEIGHTS`, `--w-*`).
  - **m1-v14** (40k steps), synthetic validation vs m1-v11: upper arm 11.8 vs 12.0, forearm 6.6 vs 6.7, wrist 16.4
    vs 16.3, head 7.5 vs 6.9 deg; thighs 10.7, shins 4.8 deg; root 1.9 cm image plane / 4.7 cm depth; up 4.3 deg.
  - m1-v14 on rendered sets (real Holistic) vs m1-v11: chest better on all sets; head 1-2.5 deg worse on
    v1/contact1/head1/pairs3; upper arm within +-2 deg except head1 (+6.5). Legs: thighs 14-40 deg, knees 16-28 cm.
    Root image plane fine (3-8 cm), depth systematically too far (ratio 1.1-1.5, median ~1.35); suspected
    apparent-size mismatch between synthetic and real landmarks, fix would go in the observation simulation (not
    done, owner to decide).
  - m1-v14 on real sessions: seated-v1 and the hands-v1 retake read seated throughout (0 switches, thigh jitter
    ~1 deg/frame); the first hands-v1 take (framed without the head) reads standing 32-40% with ~90 switches
    (framing outside the training distribution).
  - Gap calibration m1-v14 (real / synthetic p50): legs ~29 / 9 deg, root xy 5.4 / 1.4 cm, depth 41.7 / 3.0 cm
    (log ratio 0.3 / 0.0), up 6.7 / 3.6 deg.
  - mg3 first attempt went NaN at ~4500 steps with additive depth gap noise (~60 cm sd); kept as
    `data/merger/runs/mg3-nan`. Fix: depth gap as log-depth (ratio) noise along the viewing ray, depth clamp, skip
    non-finite steps.
  - **mg3** (merger on m1-v14, 20k steps), validation (includes gap noise) vs mg1: wrists 27.5 vs 27.4, forearm 13.5
    vs 13.4, head 8.0; thighs 27, shins 23.7 deg; root 6.3 / 7.4 cm; up 5.1 deg. Rendered sets: head and wrists much
    better than m1-v14 alone (head 5.8-9.4 vs 11.7-21.3 deg); thighs 2-6 deg worse than m1-v14 alone on most sets;
    depth mixed. Harness vs mg1: level on 5 of 6 sets, regression on pairs3 (hands 52.8 vs 38.4, lower arms 42 vs
    36.5). Videos `data/videos/{seated-v1,hands-v1}-mg3/`.
  - Body-contact render sets `data/m4/renders/body1`, `body2` (`render.py --events`): the generator accepted only
    belly and chest events, never thighs/lap.
  - Open: pairs3 regression, merger thigh regression, root depth bias in the simulation, framing without the head,
    combining with relational retargeting (`feature/relational-retarget`).
- 2026-11-14: live runtime / VMC output done on branch `feature/merger-runtime` (`PLAN.md` §7.6, `modules/service.md`):
  `mg3` + `m1-v14` run in `service/` via `mtcore.merged`; leg bones on by default (`--no-legs`), hip position and up as
  `/MT/Root`, `/MT/Up` in the camera frame (vspark ignores them for now). Owner's live test pending.
