# service (runtime and vspark behavior implemented 2026-11-12; face/fingers implemented 2026-11-12; learned hand model WIP)

Location: `service/` (uv project, package `mtservice`, CLI `mtservice`). Live runtime: camera or video → MediaPipe Holistic → M1 lifter (ONNX) → VRM bone rotations as VMC over UDP. Usage: `service/README.md`. Plan: `plans/m2-vspark-runtime.md` (steps 0-4 done, step 4 = vspark behavior `learned_tracker` on vspark `feature/learned-tracker`, not merged; step 5, the owner's live test, open). Decisions: `PLAN.md` §7.4. Model: `m1-v10` (`m1-v11`, body co-rotation, discarded by the owner 2026-11-12).

## Face and fingers (implemented 2026-11-12, owner's live test pending)
Plan: `plans/m2-face-hands.md` (steps 1-4 done, numbers in its Progress section); decisions: `PLAN.md` §7.5. Branch `feature/m2-face-hands` (commits 1f7f7db, ec0904d).
- Face: Holistic's own blendshape output (`output_face_blendshapes`, same model as Face Landmarker's) → 52 ARKit values (0..1), observation key `blendshapes`, sent as `/VMC/Ext/Blend/Val`. `--face-mesh full` (default) uses a cached copy of the pinned Holistic bundle with the pinned Face Landmarker bundle's full face mesh swapped in (closer to Face Landmarker than stock Holistic); `lite` = stock Holistic; `--no-face` disables. Left/Right names are passed on as MediaPipe gives them: like the bones they mirror the person (pose "left" is the person's right on the mirrored frames), as in vspark's MediaPipe Tracking.
- Fingers: `hands.py`, line-for-line port of vspark's `hand_landmarks_to_bones` (finger bones only, no wrist, from hand world landmarks), within 3e-6 deg of the TypeScript. Sent only for hands seen in that frame; `--no-fingers` disables. No smoothing, matching vspark's graph.
- Cost: about 0.7 ms/frame for face and fingers together.
- vspark side (vspark `feature/learned-tracker`, not pushed): the VMC receiver takes `Blend/Val` with ARKit names into its ARKit chain on `Blend/Apply`; other names are ignored.
- Open: finger calibration (Learned Tracking's graph has no `finger_calib`; ask after the live test).

Dependencies: `mtcore` without the `mhr` extra (numpy-only modules, `modules/mtcore.md`), mediapipe 0.10.18, onnxruntime, opencv-contrib-python. No torch, no pymomentum, no MHR assets: everything model- and skeleton-specific comes from the bundle written by `mttrain.export` (`modules/train.md`).

## Structure
| File | Role |
|---|---|
| `detector.py` | `Detector`: Holistic Landmarker set up like the recorder (same pinned `.task` model, VIDEO mode, same confidence options, mirrored 320x240 via `prepare`). Model downloaded once to `~/.cache/motiontracker/` (MD5 checked) unless `--task` is given. Reads each output stream itself: 0.10.18's Python Holistic wrapper aborts on an empty packet (no hand) and drops everything without a face (`spikes/mp-python-parity/README.md`). Also blendshapes; `model_path(face_mesh="full"\|"lite")` builds and caches the full-mesh bundle (`~/.cache/motiontracker/holistic_landmarker-float16-1+face_landmarker-float16-1-mesh.task`). |
| `sources.py` | Frame sources yielding `(t seconds, BGR frame)` at 10 FPS. `CameraSource`: reader thread keeps only the newest frame, taken every 1/fps s like the recorder's throttled loop. `VideoSource`: the recorder's stepped replay (`ts = k/fps + 1 ms`, latest frame at or before `ts`), deterministic; `realtime=True` paces it to the wall clock. `list_cameras()` (Linux: sysfs names). |
| `lifter.py` | `LiveLifter(bundle)`: checks the bundle's input layout against `mtcore.features`, then per frame `mtcore.features.StreamFeatures.push` → ONNX → `mtcore.lifter_pose.LifterPose` (restricted to the output bones, 18 of 127 MHR joints) → `(bones, confidence, present)`. `set_avatar(rest_positions | None)`, `reset()` (on camera switch). |
| `hands.py` | `solve_hands(obs)`: port of vspark's finger solve (`hand_landmarks_to_bones`) → VRM finger bone rotations. Reference: `baseline/src/hands-cli.ts` runs the frozen TS node over JSONL frames. |
| `vmc.py` | Own OSC encoder (`message`, `bundle`) and `VmcSender`. See VMC conventions. |
| `cli.py` | `mtservice` entry point and main loop; standalone or `--stdio`. `--dump-poses` / `--dump-landmarks` write JSONL for offline comparison; `--vmc off` disables sending; `--face-mesh`, `--no-face`, `--no-fingers`. |

Default bundle: `data/m1/runs/m1-v10/live/` (gitignored, CMU-derived weights; never committed).

## Stdio protocol (`--stdio`)
JSON lines; stdout carries only events (MediaPipe and log output go to stderr). Authoritative list: `cli.py` docstring.
- Commands (stdin): `{"cmd":"avatar","rest_positions":{vrmBone:[x,y,z]} | null}` (null = bundle's default avatar, AvatarSample_D), `{"cmd":"camera","index":n}` (ignored for video sources), `{"cmd":"cameras"}`, `{"cmd":"stop"}`.
- Events (stdout): `ready` (model source, bones, source, `face`: "full"|"lite"|null, `fingers`: bool), `status` (~1/s: fps, detect_ms, model_ms (includes the finger solve), latency_ms capture → output, present fraction), `cameras`, `error`, `stopped` (reason).
- Closing stdin stops the runtime, so it never outlives its parent. Unknown or malformed commands produce an `error` event, not an exit.

## VMC conventions
- One OSC bundle per frame: `/VMC/Ext/OK 1`, `/VMC/Ext/T`, `/VMC/Ext/Root/Pos` (identity), one `/VMC/Ext/Bone/Pos` per output bone, then the extensions.
- Bone names: Unity HumanBodyBones (`UNITY_NAMES`): hips, spine, chest, upperChest, neck, head, shoulders, upper/lower arms, hands, finger bones. Bones without a Unity name are skipped. Thumb: Unity's Proximal/Intermediate/Distal = VRM 1.0 Metacarpal/Proximal/Distal. `VRM_PARENT` covers the finger chains for rest offsets.
- Rotation sent as `(x, -y, -z, w)`: vspark's `rhylive_bone_mapper` applies the same self-inverse map, so vspark receives exactly the lifter's VRM normalized locals. Positions: the avatar's rest offsets from the nearest present parent (`rest_offsets`), mapped `(-x, y, z)`.
- Blendshapes: one `/VMC/Ext/Blend/Val <ARKit name> <v>` per shape, then `/VMC/Ext/Blend/Apply`.
- Extensions: `/MT/Present 0|1`, `/MT/Conf <lifter joint> <0..1>` (confidence = exp(-log_scale)). Namespace `/MT/` is provisional (`PLAN.md` §7.2, `[OPEN]`); receivers ignore unknown addresses.
- No smoothing: output goes out as is at 10 FPS. Whether vspark smooths VMC input (the model was judged through glide) is an open question in the plan.

## Checks (2026-11-12, recorded seated-v1 session video)
- Bones within 0.12 deg of offline `mttrain.infer` on the same landmarks; ONNX vs torch 3e-6.
- Per frame: detection ~21 ms (CPU), features + ONNX + VRM conversion ~1.4 ms.
- Tests (`service/tests/`, `uv run pytest`): VMC encoding against vspark's mapping (bones, Blend/Val + Apply, finger names and rest offsets); streaming vs offline inference (skips without the local bundle and session); finger port vs the TS reference on synthetic hands (`tests/fixtures`, generator `make_hands_input.py`).

## Extending
- **New frame source**: a class with `__iter__` yielding `(t seconds, BGR frame)` at the lifter's frame rate (10 FPS; the model's history offsets assume it) and `close()`; `CameraSource` also exposes `t0` for the latency stat. Wire it into `open_source` in `cli.py`. Keep the recorder's capture settings (mirroring happens in `detector.prepare`).
- **New output sink** (e.g. the deferred WebSocket): take `(t, bones, conf, present)` per frame like `VmcSender.send`, plus a rest-pose update hook if it sends positions; call it from the main loop next to `vmc.send`. New VMC extension addresses go under the same (provisional) namespace.
- **New bundle version**: `lifter.json` carries `"format": "motiontracker-lifter", "version": 1`. `LiveLifter` currently checks only the input layout (offsets, frame dim) against `mtcore.features`, not the version. When the bundle format changes (new inputs such as hands/face, other output bones), bump `version` in `mttrain.export` and branch on it in `LiveLifter`; feature changes go into `mtcore.features` so training, export and runtime stay identical.
- **New output bones**: come from the bundle (`output_bones`); add their Unity names to `vmc.UNITY_NAMES` and parents to `vmc.VRM_PARENT`.
- **Learned hand model** (WIP, plan `plans/m4-hand-model.md`): runner planned next to `hands.py`, selectable by flag; replaces `hands.py` behind the same per-frame `{vrmBone: quat}` output for the hands seen; finger names/parents already exist in `vmc.py`.
- **Learned face path** (`PLAN.md` §3.11/§7.5): may replace the MediaPipe blendshapes or consume them (e.g. as merger input); keep the `{ARKit name: 0..1}` observation key so `vmc.py` is unchanged.

## Open
- Owner's live test of face and fingers; finger calibration; learned hand model (WIP, `plans/m4-hand-model.md`).
- Checking the stream with vspark's existing `vmc_receiver` (plan step 3) is not recorded as done; only the unit test against its mapping exists.
- vspark-side smoothing, camera naming (OpenCV indices vs the browser's device list), camera exclusivity with MediaPipe Tracking, the OSC namespace name (plan open questions).
