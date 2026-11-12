# Plan: face blendshapes and fingers in the live runtime

> Branch: `feature/m2-face-hands` (from `feature/m2-vspark-runtime`); vspark `feature/learned-tracker` · Status: steps 1-4 done (2026-11-12); step 5 (owner live test) open
> This plan is the seed context for a worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

Learned Tracking in vspark drives the face (52 ARKit blendshapes) and the fingers too, from the
MediaPipe Holistic run the runtime already does, so it can replace vspark's MediaPipe Tracking as a
whole. This slice uses no new learned models: MediaPipe's blendshape model for the face, a port of
vspark's finger solve for the hands. A learned hand model replaces the port afterwards (own plan).
Decisions: `PLAN.md` §7.5.

## Decisions

- Face: MediaPipe's blendshape model run on Holistic's face landmarks; no second face detector.
  `[USER]` (2026-11-12) Revisit when the merger is built: it may want blendshapes as input. `[USER]`
  The Holistic bundle ships the same model and Holistic's Python API runs it (`output_face_blendshapes`),
  so no extra interpreter is needed.
- Face mesh: Face Landmarker's full mesh swapped into the Holistic bundle by default, `--face-mesh lite`
  for stock Holistic; flag only, no vspark setting. `[USER]` (2026-11-12) See `PLAN.md` §7.5.
- Blendshapes keep MediaPipe's left/right, which mirrors the person like the body does (pose "left" is
  the person's right on the mirrored frames). `[CLAUDE]` (an earlier swap was wrong, see `PLAN.md` §7.5)
- Hands: port vspark's `hand_landmarks_to_bones` first (identical to the frozen copy in
  `baseline/vendor/backend/signal/nodes/hand_landmarks_to_bones.ts`), learned hand model after. `[USER]` (2026-11-12)
- Body stays m1-v10. `[USER]`
- Output over the same VMC stream: `/VMC/Ext/Blend/Val` (ARKit names) + `/VMC/Ext/Blend/Apply`;
  finger bones as `/VMC/Ext/Bone/Pos` with Unity names. `[CLAUDE]`
- vspark's VMC receiver gets standard `Blend/Val` support (today it reads ARKit only from RhyLive's
  `/Face`), feeding the same `arkit` event and mapper chain. `[CLAUDE]`

## Constraints

- No new dependency without asking the owner, and every model/dependency in `docs/LICENSES-REGISTER.md`
  first. Blendshape model: Apache 2.0, verified from its model card (2026-11-12).
- Latency: blendshapes and fingers together well under 2 ms/frame.
- vspark conventions (its CLAUDE.md): i18n en/de and help for any UI change, `vs-` handles, lint and
  tests before e2e, controls manifest.

## Files in scope

- `service/src/mtservice/`: detector (face landmarks → blendshape model input), a `face.py`
  (blendshapes), a `hands.py` (finger port), `vmc.py` (Blend/Val, finger bones), `cli.py` (flags to
  turn face/fingers off, status fields), tests.
- `docs/LICENSES-REGISTER.md`, `dev-notes/`.
- vspark: `vmc_receiver/manager.ts` (`Blend/Val`), possibly the learned_tracker graph (finger
  calibration, see open questions), tests, help text.

## Out of scope

- Learned face head, learned hand model, merger, gaze beyond the blendshapes' eyeLook values.
- Head rotation from the face mesh (the body model keeps driving the head; vspark's mixer can mix).

## Approach

1. **Blendshape model.** Pin `face_landmarker.task` (MD5 in the register), extract the blendshape
   model, verify its license. Input: the 146-landmark subset MediaPipe's blendshape graph uses, in
   its normalization. Parity: run MediaPipe Face Landmarker (with blendshapes) and our path on the
   session video; compare per shape (target: mean abs difference small next to the shapes' range,
   reported with numbers). Holistic's face landmarks vs Face Landmarker's own also differ: report both.
2. **Finger port.** Python port of `hand_landmarks_to_bones`; parity against the TypeScript baseline on
   the session's recorded hand landmarks (< 0.1 deg). Wrist/hand bone stays from m1-v10; the port gives
   finger bones only (local to the hand). Note the forearm-twist mismatch for the merger.
3. **Runtime + VMC.** Wire both in, flags to disable each, timing in the status event.
4. **vspark.** `Blend/Val` in the VMC receiver; check that fingers and blendshapes reach the avatar
   (e2e on the session video like the body); help text update.
5. **Owner live test.**

## Open questions

- Finger calibration: vspark's MediaPipe Tracking has a `finger_calib` node (hold the hand in the
  avatar's rest); does Learned Tracking need it in its graph?
- Does the port's smoothing (if any) belong in the runtime or stay in vspark?

## Acceptance / verification

- Blendshape parity numbers and finger parity numbers reported; runtime tests pass.
- Added time per frame for face + fingers reported.
- vspark lint/tests pass; with Learned Tracking on, face expressions and fingers move the avatar.

## Output

Commits on `feature/m2-face-hands` and vspark `feature/learned-tracker`; report to the owner before
merging.

## Progress (2026-11-12)

- Step 1 done: blendshapes in the runtime (`detector.py`: `--face-mesh full|lite`, `--no-face`;
  `vmc.py`: `Blend/Val` + `Blend/Apply`). On the session video, against Face Landmarker on the same
  frames (662 frames): full mesh mean abs 0.018, median correlation 0.87 over 33 active shapes; lite
  0.028 / 0.72. Body output bit-identical with full, lite and no face.
- Step 2 done: `hands.py`, a line-for-line port of the finger solve; `baseline/src/hands-cli.ts` runs the
  TypeScript node for reference. Port vs TypeScript on the same landmarks: worst 3e-6 deg (session, 10,365
  bones; committed test on synthetic hands). The solve itself moves up to 0.58 deg (p99 0.12) when the
  landmarks are rounded to 5 decimals (10 µm): a measure of its sensitivity, relevant to its jitter.
  No smoothing: vspark's graph has none on the hands either.
- Step 3 done: fingers and blendshapes in the runtime (`--no-fingers`, `--no-face`); the `ready` event
  reports `face` and `fingers`. Finger bones go out only for hands seen this frame. Cost on the session
  (guarded, cores 8-15): detect 21.1 vs 20.5 ms, model + fingers 1.92 vs 1.77 ms per frame, so about
  0.7 ms for both.
- Step 4 done (vspark `feature/learned-tracker`, fe18fbe, 6db9016; not pushed): the VMC receiver collects
  `Blend/Val` with ARKit names and feeds them to the ARKit chain on `Blend/Apply` (other names ignored);
  help and kind descriptions (en/de). vspark lint and all tests pass. End to end (runtime on the session
  video into vspark's real VmcManager and Learned Tracking graph): 670 pose and 667 blendshape publishes,
  all 30 finger bones, ARKit mapped to the avatar's expressions (eye close, mouth shapes, brows).
- Open: finger calibration (vspark's MediaPipe Tracking has a `finger_calib` capture; Learned Tracking's
  graph has none). Not needed for fingers to work; ask the owner after the live test. Smoothing: none,
  matching vspark's graph.
- Corrected: the left/right swap of step 1 was wrong (face mirrored the other way from the body);
  removed. vspark's MediaPipe Tracking was right not to swap.
