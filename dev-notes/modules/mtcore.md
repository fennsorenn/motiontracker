# mtcore (implemented)

Location: `core/` (uv project, Python package `mtcore`). Shared code moved out of the M3 spike (`modules/mhr-vrm-spike.md`) for M1 and later stages; consumers: the spike, `train/` (`modules/train.md`) and the live runtime `service/` (`modules/service.md`). Usage: `core/README.md`.

## Using it
- Consumers add it as an editable path dependency: `mtcore = { path = "../core", editable = true }`.
- Consumers choose their own torch build (CPU in the spike, CUDA in `train/`). Core's own CPU torch is a dev-only dependency for its tests.
- The base install depends on numpy only. pymomentum (imported by `mhr` and `cmu_fit`) is the optional extra `mtcore[mhr]`, installed by `train/` and the M3 spike; `service/` installs mtcore without it and uses only the numpy-only modules (`features`, `lifter_pose`, `convert`, `vrm`, `paths`, `face_filter`).
- Data paths come from `paths.py`; `MT_DATA` overrides the gitignored `data/`.

## Modules
| Module | Role |
|---|---|
| `paths.py` | Repo and data locations. |
| `mhr.py` | Loads the MHR rig via pymomentum **without blendshapes** (`MHR.from_files` segfaults with them, pymomentum 0.1.114 CPU), FK (`skeleton_state`, `positions`), `[Limits]` parser (`limits`), `FLEX_T` (flexible-translation parameters, kept at zero). |
| `convert.py` | MHR skeleton state → VRM 1.0 normalized local rotations (`MhrToVrm`), VRM FK, quaternion helpers. `REST_ALIGNED` (torso + shoulders) map rest-to-rest, other bones use direction matching. Method: `modules/mhr-vrm-spike.md` (Conversion). |
| `vrm.py` | Minimal VRM 0.x/1.0 humanoid reader (0.x turned to face +Z). `load_vrm_mesh(path, skip=("Hair",))` (WIP, `relational-retarget`): rest-pose skinned vertices (world metres, same +Z facing as `load_vrm`) and the humanoid bone per vertex (strongest joint, or its nearest humanoid ancestor); meshes/joints named "Hair" skipped since the human mesh has none. |
| `cmu.py` | CMU ASF/AMC reader + FK (cm), with optional per-frame rotations. |
| `cmu_fit.py` | CMU clip → MHR (pymomentum sequence IK). See below. |
| `landmarks.py` | `LandmarkModel`: MediaPipe's 33 pose landmarks from an MHR pose. See below. |
| `features.py` | M1 model input (moved from `mttrain.features`, 2026-11-12), numpy only. See below. |
| `lifter_pose.py` | M1 lifter output → MHR globals → VRM locals, numpy only. See below. |
| `face_filter.py` | Causal blendshape post-processing (eye correction, glide), plain Python. See below. |
| `facepose.py` | Head pose from Holistic's face mesh: weighted similarity fit of MediaPipe's canonical face model (33 Procrustes points); metadata read from the cached Face Landmarker bundle with a minimal protobuf wire decoder; numpy only. Used by the merger (`modules/merger.md`). |
| `relretarget.py` | WIP prototype (2026-11-14, branch `feature/relational-retarget`; not a decision). See below. |

## cmu_fit
- `fit(subject, clip, ...)`: spike-style single-clip fit (shared scales + per-frame pose), default `PAIRS`.
- `fit_subject_scales(ch, subject, clips, ...)`: one set of performer scales per CMU subject from frames sampled across its clips.
- `fit_clip(ch, subject, clip, scales, ...)`: per-frame pose for one clip with fixed subject scales, solved in chunks.
- `PAIRS_M1` (default for the M1 functions): `PAIRS` with loosened spine/neck weights (`LOOSE_SPINE`; spike finding: spine scales leaned on their limits) plus thumb pairs (constrain hand roll, hence forearm twist).
- Head orientation constraint from CMU's head bone (`HEAD_WEIGHT`): positions alone leave head orientation free. Without it the first M1 model never moved the head.
- Neck/head split (2026-11-08, fits `fits-v3`):
  - `head_offset`: per-subject CMU head calibration offset (mean thorax-to-head rotation), removed from the head target.
  - `head_post`: `fit_subject_scales(..., head_post=True)` returns `(scales, post)`; `post` makes the subject's mean head pose relative to `c_spine3` equal MHR's neutral.
  - `NECK_PRIOR = 0.1`: Motion prior on the neck/head parameters in `fit_clip`.
  - Effect: head joint at its limits in 1-4% of target frames (was 23-77%). `mttrain.prep_fits` stores both per subject (`modules/train.md`).

## landmarks
`LandmarkModel` computes MediaPipe's 33 pose landmarks from MHR joints (shoulders, elbows, wrists, hand points, hips, ...) and from mesh vertices (face points, etc.) that are carried rigidly with their joint, so no full mesh skinning is needed per frame. MediaPipe's systematic placement differences from MHR anatomy are not modelled here; that is `mttrain.mpstyle` (`modules/train.md`).

## features
Shared by training (`mttrain.dataset`, `model`, `infer`, mirrored by `gpugen/features.py`), export and the live runtime, so they cannot diverge.
- Batch path: `causal_scale` (shoulder-width scale over `SCALE_WINDOW`, `DEFAULT_SCALE` before any pose), `frame_features` / `scaled_features` (`FRAME_DIM` per frame; `head3d=False` masks the 3D head points), `history_indices` (samples at `OFFSETS` 0-1.6 s back), `model_input`.
- `StreamFeatures(head3d)`: one frame at a time (`push(t, world, image, vis, present)` → model input for that frame), identical to the batch path (`tests/test_features.py`).
- Changing anything here changes the model input: retrain, re-export, and keep `gpugen/features.py` in sync.

## lifter_pose
`LifterPose(out_joints, out_parent_idx, joint_names, parents, rest_state, vrm_rest_positions, bones=None)` chains the lifter's rest-relative rotations (camera axes, `R0` → MHR world) into MHR world rotations (unpredicted joints keep their rest pose relative to the parent) and converts them with `MhrToVrm` (`mhr_globals`, `vrm`).
- Needs only a small MHR table (`table()` / `from_table()`: joint names, parents, rest state), so no pymomentum or MHR assets. `mttrain.export` writes the table into the live bundle.
- `bones=`: compute only the MHR joints those VRM bones need (plus forearm twist joints); live use computes 18 of 127 joints, ~1 ms/frame instead of ~3 ms. `set_avatar` swaps the avatar rest pose.
- Also `quat_to_mat`, `mat_to_quat` (xyzw); `mttrain.skeleton` re-exports `quat_to_mat`.
- Used by `mttrain.infer.Runner` (same output as before within ~0.01 deg, float32 vs float64 rest data) and `mtservice.lifter.LiveLifter`.

## relretarget (WIP prototype)
Relation-preserving retargeting onto a VRM avatar ("option A", `dev-notes/ARCHITECTURE.md` retargeting section), numpy-only. Prototype built 2026-11-14 (commit c5d7b7d, branch `feature/relational-retarget`); awaiting the owner's evaluation, not adopted.
- **Anchors**: torso bones (hips..head) and the other arm's upper/lower arm. Each has a frame, a bone axis (in frame coordinates) and a `Profile`: surface radius per position along the axis and direction (90th percentile of skinned vertices, kernel-interpolated). Human profiles from the MHR identity mesh, avatar profiles from `vrm.load_vrm_mesh`.
- **Source frames must match rotation copy**: human anchor frames are `MhrToVrm.globals` times the avatar's rest frame, i.e. exactly what rotation copy gives the avatar. Independent anatomical frames from each body's rest geometry failed, because the converter's per-bone rest alignment and forearm twist differ between bodies.
- **Hand points**: palm centre (plus index/middle/thumb distal joints with `--tips`).
- **Relations**: per anchor (u along the axis, direction, signed gap to the surface, absolute) and hand-hand vectors in the upperChest frame. Weights `(1 - s/falloff)^2 / (s + 2 cm)^2`; falloff 30 cm (body), 60 cm (hand-hand); rotation-copy fallback weighted like an anchor at 25 cm. All hand-tuned.
- **Solve**: two-bone arm IK keeping the copy's elbow plane and the hand's global rotation, 6 rounds; relations to moving targets (other hand, other arm) take half steps (full steps oscillated).
- **Demo**: `uv run python -m mttrain.retarget_demo <render dir> --out-dir <dir> [--avatar AvatarSample_F] [--tips]` (in `train/`): true MHR pose and identity mesh from `seq.npz` -> `relational.poses.jsonl`; prints per segment palm-palm distance, signed left-right separation, palm gap to head surface and closest hand-hand pair for true body / rotation copy / relational. Results are in the ARCHITECTURE retargeting section; videos (gitignored) in `data/videos/retarget-*`.
- **Open**: weights hand-tuned; only arms move (no torso lean when out of reach); legs not anchors (the ground truth has no legs); the source body here is the true identity, live use would need a mean or estimated body; runtime cost not measured; not connected to the merger output; whether it stays in `mtcore` if adopted (vs "option B", retargeting inside the merger, `modules/merger.md`).

## face_filter
Algorithmic, causal post-processing of MediaPipe's ARKit blendshapes (owner decision 2026-11-14, `PLAN.md` §3.11/§7.5; learned face head in reserve). Used by `mtservice.face` (live) and `mtservice.faceproc` (offline), `modules/service.md`.
- `EyeFilter`, per eye:
  - Open-eye blink level fitted online as a linear function of that eye's `eyeLookDown` (exponential forgetting, tau 60 s; prior a=0.18, k=0.4); frames above fit + 0.12 are treated as blinks and excluded from the fit.
  - Blink as an event (fix f9dce9a: MediaPipe at ~10 Hz catches blinks part-way and unevenly): the excursion above the fitted open level is measured against the open-eye noise (residual sd, tracked online, prior 0.06). Below 2 sd (`BLINK_DEADZONE_SD`) the eye reads open; it reaches fully closed at 3.5 sd (`BLINK_EVENT_SD`), smoothstep in between.
  - Both eyes blink from their mean excursion (noise sd / sqrt 2), unless it is a wink: one eye closed while the other stays below 0.5 sd (`BLINK_WINK_SD`) above its open level.
  - Squint rescaled above a resting baseline that follows drops fast (1 s) and rises slowly (20 s).
  - Squint left/right linked to their mean when they differ by < 0.15, independent above 0.35 (`LINK_START`/`LINK_FULL`; blink no longer uses this linking).
- Glide: smoothing-free upsampling like the harness's bone glide (`mteval.upsample`): linear from the shown value to the new one, finishing one tick before the next detector frame is expected (mean of the last 10 intervals).
- Effect on the recorded sessions (before the f9dce9a blink fix): resting blink 0.27-0.34 → 0.02-0.06, squint 0.39-0.40 → 0.12-0.13, left/right difference 0. After the fix, blinks reaching >= 0.9 closed: seated 28 of 41 events, hands 35 of 55 (before: 0); rest stays at 0. Seated blink rate ~37/min, possibly some false blinks (to watch live).
- Status: merged into `dev` (2026-11-14), used live via `mtservice` (`modules/service.md`).
- Tests live with the consumer: `service/tests/test_face.py`.

## Tests
`core/tests/` (`uv run pytest`): streaming vs batch features, conversion round trip (direction match for non-`REST_ALIGNED` bones; MHR rest → identity locals and torso following MHR world rotation for `REST_ALIGNED`; arm world rotations equal to all-direction matching), CMU fit, landmarks, quaternions. Tests skip when `data/mhr`, `data/cmu` or the avatar files are missing.

## Extending
- New reusable code that more than one project (spike, `train/`, later the service) needs goes here, not into a consumer.
- Keep torch out of the runtime dependencies; consumers pin their own build.
- Anything the live runtime needs must stay numpy-only and must not import `mhr` / pymomentum at module level; pymomentum-dependent code goes behind the `mhr` extra.
