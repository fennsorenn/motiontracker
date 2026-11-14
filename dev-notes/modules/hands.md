# hands (`mttrain.hands`, M4 learned hand model: WIP)

Location: `train/src/mttrain/hands/` (part of the `train` uv project, `modules/train.md`). Learned hand model: MediaPipe hand landmarks → finger parameters (MHR's finger joints) and, in v2, each hand's orientation and position in the camera frame. Plan, numbers and open items: `plans/m4-hand-model.md`; decisions `PLAN.md` §5.6. Data, renders and runs live in gitignored `data/m4/` (`pool/`, `calib/`, `renders/`, `runs/<name>/`).

Status (2026-11-13): v1 single-hand model (`h1`, `h2`), v2 pair model (`p2`) and v3 pair model (`p3`) trained and evaluated on renders; combination onto the body model's arm done offline (`wrist`). v3 (counting, rests, body contact, detector scores) implemented, see "v3" below. Not yet in the live runtime (`service/`, `modules/service.md`). Results for p2: plan section "Pair model p2".

## Shared building blocks
- `skeleton`: MHR hand skeleton, MediaPipe's 21 points ↔ MHR joints, finger parameters, FK. Canonical hand = MHR's left hand; right hands are mirrored across x (one model for both).
- `motion`: procedural finger motion (gesture library, coupled random poses, holds, minimum-jerk transitions, tremor, taps).
- `stage`: hands-in-use arm choreography by IK for rendered single-hand sessions (`render prepare --hands`).
- `pairs`: two-hand interaction choreography (palms, steeple, tap, count_on_hand, rub, clap, cross, occlude, approach, palm turns, wrist bends); used for renders (`render --pairs`) and the v2 pool.
- `collide`: interpenetration check on MHR's own collision capsules (arm vs body, hand vs hand); since v3 also an exact body mesh distance (`BodyMesh`).
- `relax`: mild collision physics on authored motion (differentiable `collide` capsules through `gpugen.fk.MhrFK`, pull back to the intended pose).
- `keysheet`: review sheets and clips of interaction key poses (Blender, two views), for owner review before rendering whole sets.
- Input features: `mtcore.hand_features.canonical` (shared with the runtime; `modules/mtcore.md`).

## Calibration on rendered hands (what the synthetic observations reproduce)
- `calibrate`: MediaPipe world landmarks vs truth in the canonical palm frame (bias, noise, mirroring, detection rate) → `--json` for `observe`.
- `imagecal`: image-space landmarks → camera-frame hand orientation vs truth (Holistic or Hand Landmarker via `--landmarks`).
- `failures`: identity swaps, doubled hands, palm/back flips per detected hand.

## v1: single-hand model
`pool` (procedural finger sequences) → `observe.Observer` (synthetic MediaPipe observations) → `model.HandNet` (8 frames at 10 Hz → finger parameters) → `train` → `evaluate` (finger segment directions in a palm frame vs truth; `--system ported|<ckpt>`) / `infer` (`LearnedHands`, poses with learned fingers for avatar videos).

## v2: pair model
- `pool2`: posed two-hand sequences (arms + fingers, from `pairs` or `stage`) with the body landmarks the GPU generator needs. `--name <stem>` writes a shard; `--merge <stems...> --name <out> --n 0` merges shards (no generation; `--n` is required, so pass 0).
- `observe2`: synthetic observations of both hands (image and 3D landmarks, camera) with flips, swaps, doubled and lost hands, more when the hands are close; mixed over both detectors with no detector flag. Passes camera-frame truth points as `truth["cam"]`.
- `model2.PairNet`: both hands' recent history → the own hand's fingers, orientation and position; runs once per hand, the right hand as the mirrored scene, one set of weights.
- `train2 --name <run>`: orientation target is the true 3D palm frame in the camera frame (since p2).
- `evaluate2 <set> --ckpt <run>/best.pt [--landmarks landmarks.jsonl|landmarks-hl.jsonl]`: fingers and camera-frame orientation against exact truth.

## Combination onto the body model (`wrist`, merger stand-in)
To be replaced by the learned merger (WIP, `modules/merger.md`), whose base pose is this combination done in the body model's frame (`merger.batch.combine`).
`HandOrient` chains the pair model's palm frame (camera frame) → body model frame (`lifter_pose.R0`) → MHR wrist via the rest palm frame → VRM hand global via `MhrToVrm` (`q_rest_inv`, `D`; `mtcore.convert`) → local to the body model's `lowerArm`.
- The whole rotation goes onto the hand bone: the rendered truth carries the forearm twist on the hand. `twist_to_forearm=True` moves the twist onto the lower arm but put it 129 deg off the truth; it stays off until a truth convention for forearm twist exists.
- Bends beyond `MAX_BEND_DEG` (100) after removing the forearm twist are rejected (likely palm/back flip): the body model's hand is kept.
- `pair_track` / CLI: `python -m mttrain.hands.wrist <ckpt> <landmarks> --body <poses> --out <out> --avatar <name> [--no-wrist]` writes a pose track with learned fingers and (unless `--no-wrist`) learned hand orientation. Since v3, no hold: the result blends to the body model's hand by the pair model's confidence (rotation scale).

## v3 (implemented, 2026-11-13)
Training data for counting, off-screen rest and body contact, plus a per-hand detector score input; run `p3`. Numbers: plan sections "v3 progress" and "Pair model p3" in `plans/m4-hand-model.md`; decisions `PLAN.md` §5.6 (v3).
- `motion`: independent per-finger poses, thumb-first counting, counting routines, `relaxed_pose(flat=)`.
- `stage`: rest episodes (`p_rest`; rests incl. hanging beside the chair; `with_rest` weights, `fix_rest`, clear callback).
- `collide`: `BodyMesh`, exact signed point-to-triangle distance to MHR's skinned mesh with the arms removed; `set_pose(m, sides)` can keep the other arm so a hand can rest on the other hand; `closest` / `normal` / `smooth_normal` / `surface`; `closest_on_triangles`. `seg_closest` fixed (it measured to capsule start points only).
- `pairs`: body contact events (thighs, lap_folded, belly, chest). `contact_spec` places the palm heel first and lays MHR's palm surface on the skin chord (measured palm pads); `settle` drops the palm pads onto the skin after the arm IK; `drape` / `drape_thumb` / `curl_thumb_tip` for fingers; `resting_hand`; `fix_rest` keeps rest arms out of the body; relaxed elbow cost; `thigh_elbow` (forearms along the thighs for folded hands); `knees_together`; mesh checks `mesh_clear`, `mesh_clip_frames`; per-event style dict.
- `relax`: body mesh penalty (`Relaxer.mesh`), keeps hands out of the body mesh.
- `keysheet`: `--wide`, `--knees-together`, `--prefer-pass`, level side view, palm pad diagnostics.
- `pool2`: `stage_sequence`, rejects sequences with mesh clipping, `--dir`, resumable (partial files; `--fresh` restarts).
- `scorecal` (new): calibrates the detector score per detector and state on renders (Holistic pose visibility of the wrist/hand points, Hand Landmarker handedness score).
- `observe2` / `model2` / `train2`: detector score input slot (synthesized from `scorecal`), out-of-view hands trained, confidence outputs; `train2 --resume`.
- `wrist`: confidence blend to the body model's hand, no hold.
- `evaluate2`: per-finger, out-of-view and model-scale reporting.
- Render tools: `tools/render/handlandmarker.py` stores scores and is resumable; `render_blender.py` resumes per frame; `render.py prepare` is resumable. Avatar-video renders (`tools/avatar-video`) are not resumable yet.

Owner decisions on the way: the folded-in-the-lap pose stays at its best version although the top hand sits a fist's height above the leg (refine later); clipping correction data goes to the merger, not the hand model (deferred).

Merger support (2026-11-14, `modules/merger.md`): `pool2 --full` (full MHR parameter sequences) and `--no-tremor`; `HandMotion` tremor flag; `observe2.PairObserver` optional camera (shared with the body observation); `evaluate2.LearnedPair` exposes raw outputs (`raw`, `lost`, `score`).

## Extending
- Anything inference needs must stay in `mtcore.hand_features` so training and the runtime cannot diverge.
- New detector effects: measure them first (`calibrate`, `imagecal`, `failures`) on renders, then add them to `observe`/`observe2`.
- New interaction moves go into `pairs`; check them with `collide` and `keysheet` before rendering a set.

## Open
See the plan ("Pair model p2" Open, "Pair model p3"): occlude/wrist/turn acceptance, finger fixes and the single-hand finger regression in p2 (p3 level on hands2), forearm twist convention, Hand Landmarker runtime option, runtime integration. From p3: Holistic fingers worse than p2 on pairs3; out-of-view finger truth inconsistent between generators (pair sequences start flat, stage rests relaxed); folded-lap pose refinement; clipping correction in the merger; resumable avatar-video renders.
