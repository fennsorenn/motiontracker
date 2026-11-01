# Plan: M3 spike, MHR ↔ VRM 1.0 mapping

> Branch: `feature/m3-mhr-vrm-spike` · Status: draft
> This plan is the seed context for a cloud worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

Confirm that MHR works as the canonical skeleton for VRM 1.0 output (PLAN.md §3.10, §9 #1),
before M1 trains in MHR space (owner decision 2026-11-01). Answer three questions:

1. Do MHR rotations map onto VRM 1.0 humanoid bones with a fixed, thin conversion?
2. Can MHR identity + scale parameters represent the VRoid sample avatars' stylized proportions?
3. Do spatial relations (hand-to-face, hand-to-hand) survive the MHR → VRM conversion?

## Constraints

- MHR is the canonical skeleton; VRM poses derive from MHR rotations `[USER]`.
- Performer and avatar are separate MHR identities `[USER]`.
- Licensing: MHR code and assets Apache 2.0, pymomentum MIT, CMU data must never be committed.
  VRoid samples: only the eleven permissive ones in the license register; AvatarSample_M is excluded.
- Assets and datasets live in gitignored `data/` (`data/mhr/assets/`, `data/cmu/all_asfamc/`).
- The Python environment for MHR (PyTorch + pymomentum) is an owner decision; see Open decisions.

## What is already known (from `data/mhr/assets/compact_v6_1.model`, no code run)

MHR skeleton: `root`, spine `c_spine0..3`, `c_neck`, `c_head`, per side clavicle, upper arm,
lower arm, `wrist_twist`, wrist, thumb `0..3`, fingers `1..3`, legs with upper/lower leg, foot,
talocrural, subtalar, transverse tarsal, ball; procedural twist joints (`*_twist*_proc`) and
null joints for eyes and fingertips.

Bone lengths are explicit parameters: `scale_uparms`, `scale_lowarms`, `scale_shoulder_width`,
`scale_spine_length`, `scale_neck_length`, `scale_hip_width`, `scale_uplegs`, `scale_lowlegs`,
`scale_r/l_hands` and per-finger-segment lengths and offsets. Upper and lower arm scale
independently, which the VRoid proportions need (upper arm ≈ forearm, or shorter). The file
also has a `[Limits]` section (joint limits), useful for the impossible-pose metric.

Draft mapping (to verify):

| VRM 1.0 bone | MHR joint(s) | Notes |
|---|---|---|
| hips | root | Check root = pelvis, not ground. |
| spine / chest / upperChest | c_spine0..3 | 4 MHR segments onto 2–3 VRM bones; compose rotations. |
| neck / head | c_neck (+ neck twist procs) / c_head | |
| left/rightShoulder | l/r_clavicle | |
| upperArm | uparm (+ uparm twist procs) | |
| lowerArm | lowarm (+ lowarm twist procs) | VRM has no twist bones: fold twist into lowerArm/hand. |
| hand | wrist_twist ∘ wrist | |
| thumbMetacarpal / Proximal / Distal | thumb0..3 | 4 MHR joints onto 3 VRM bones; decide which pair composes. |
| index/middle/ring/little Proximal / Intermediate / Distal | index1..3 etc. | MHR `pinky` = VRM `little`. |
| upperLeg / lowerLeg | upleg / lowleg | |
| foot / toes | foot ∘ talocrural ∘ subtalar ∘ transversetarsal / ball | |
| left/rightEye | l/r_eye_null | Gaze output. |

## Findings so far (2026-11-01)

Code: `spikes/m3-mhr-vrm/` (uv, CPU PyTorch). `dump_skeleton.py`, `fit_avatars.py`, `convert.py`,
`test_roundtrip.py`, `test_positions.py`.

- **pymomentum crash:** `MHR.from_files` segfaults in `Character.with_blend_shape`
  (pymomentum-cpu 0.1.114, torch 2.8 CPU). Not needed: the skeleton never depends on identity
  blendshapes. The spike loads the rig with pymomentum directly.
- **Identity vs scale:** MHR's 45 identity coefficients only shape the mesh. Bone lengths come from
  the `scale_*` model parameters, which are additive (1 unit ≈ 10 cm for limb segments). PLAN.md's
  "identity parameters give bone lengths" was imprecise.
- **Rest pose:** cm, y up, +x = character's left, z forward; A-pose with bent elbows. Root = pelvis.
- **Avatar fit (uniform scale + MHR scale parameters, segment lengths):** spine, neck, upper arm,
  forearm, hands and hip width fit all eleven avatars within MHR's human limits. Shoulder width
  needs ~2.4× MHR's narrowest setting for the female-type VRoid bodies; lower legs reach or pass
  the limit on several avatars. Clavicle length and head-to-eye distance have no parameter.
  MHR limits are solver penalties, not FK limits, so out-of-range values work geometrically.
- **Flexible translations:** pose parameters `*_flexible` (spine/neck length, shoulder width, arm,
  hip, leg) translate joints. VRM bones only rotate, so these must stay zero (or be absorbed).
- **Conversion:** `G_vrm(b) = Q_mhr(j) · Q_mhr_rest(j)⁻¹ · D(b)`, D = rest-direction alignment.
  With flexible translations at zero, bone directions match to < 0.05° for every bone on two
  avatar builds (200 random poses). End-effector error on a VRM skeleton with MHR proportions,
  relative to hips, from folding joints VRM lacks:

  | Pose range (fraction of MHR limits) | Head/hands mean (max) | Thumb tip | Foot |
  |---|---|---|---|
  | 25% | 0.4 cm (0.8) | 0.6 cm (1.3) | 1.1 cm (1.8) |
  | 50% | 0.8 cm (1.6) | 1.2 cm (2.7) | 1.3 cm (2.6) |

  The head/hand error comes from MHR's extra spine joint (`c_spine0`). Moving forearm twist from
  the hand onto lowerArm (driving it by `wrist_twist`) adds 0.4–0.8 cm at the hand because the
  twist axis isn't exactly the forearm axis; a swing-twist split would avoid that.

## Approach

1. Load MHR (LOD 1 is enough), dump the rest-pose skeleton: joint hierarchy, positions,
   orientations, scale parameter effects.
2. Build the mapping table above for real, including rest-pose differences (MHR rest pose vs VRM
   T-pose with identity rotations) and the composition rule for merged joints.
3. Fit each VRoid sample avatar (unsuffixed, permissive) as an MHR identity: match VRM humanoid
   joint positions with identity coefficients + scale parameters. Report residual per joint.
4. Round trip on CMU clips: CMU ASF/AMC → MHR pose (performer identity) → VRM rotations →
   VRM forward kinematics. Compare joint directions and spatial relations (hand-face, hand-hand
   distances scaled to body size) before and after.
5. Write findings into PLAN.md §3.10 and §9 #1, recommend go / fallback (own VRM-superset
   skeleton, MHR only for rendering).

## Out of scope

- The retargeting layer itself (M4), training anything (M1), VRM 0.x → 1.0 avatar conversion
  beyond what the fit needs, rendering.

## Acceptance / verification

- Mapping table covers every VRM 1.0 required humanoid bone, with a tested conversion function.
- Avatar fit residuals reported per avatar and joint; owner decides if they're acceptable.
- Round-trip numbers on a handful of CMU clips (upper-body-heavy: gestures, reaching).

## Open decisions

- ~~Python environment~~: uv + pip, CPU PyTorch for the spike (owner, 2026-11-01).
- Allow avatar fits outside MHR's human limits (avatar side only; performer stays within)?
- Flexible translation parameters: keep at zero in the canonical pose?
- Forearm twist on VRM hand (as now) or split onto lowerArm?
- CMU round trip (step 4) needs a CMU → MHR fit, which is also M1's data preparation.

## Output

Findings in PLAN.md; code under a spike directory; merge into `dev`.
