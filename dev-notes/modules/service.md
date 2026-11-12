# service (runtime implemented 2026-11-12; vspark behavior planned)

Location: `service/` (uv project, package `mtservice`, CLI `mtservice`). Live runtime: camera or video → MediaPipe Holistic → M1 lifter (ONNX) → VRM bone rotations as VMC over UDP. Usage: `service/README.md`. Plan: `plans/m2-vspark-runtime.md` (steps 0-3 done; step 4, the vspark behavior `learned_tracker`, and step 5, the owner's live test, open). Decisions: `PLAN.md` §7.4.

Dependencies: `mtcore` without the `mhr` extra (numpy-only modules, `modules/mtcore.md`), mediapipe 0.10.18, onnxruntime, opencv-contrib-python. No torch, no pymomentum, no MHR assets: everything model- and skeleton-specific comes from the bundle written by `mttrain.export` (`modules/train.md`).

## Structure
| File | Role |
|---|---|
| `detector.py` | `Detector`: Holistic Landmarker set up like the recorder (same pinned `.task` model, VIDEO mode, same confidence options, mirrored 320x240 via `prepare`). Model downloaded once to `~/.cache/motiontracker/` (MD5 checked) unless `--task` is given. Reads each output stream itself: 0.10.18's Python Holistic wrapper aborts on an empty packet (no hand) and drops everything without a face (`spikes/mp-python-parity/README.md`). |
| `sources.py` | Frame sources yielding `(t seconds, BGR frame)` at 10 FPS. `CameraSource`: reader thread keeps only the newest frame, taken every 1/fps s like the recorder's throttled loop. `VideoSource`: the recorder's stepped replay (`ts = k/fps + 1 ms`, latest frame at or before `ts`), deterministic; `realtime=True` paces it to the wall clock. `list_cameras()` (Linux: sysfs names). |
| `lifter.py` | `LiveLifter(bundle)`: checks the bundle's input layout against `mtcore.features`, then per frame `mtcore.features.StreamFeatures.push` → ONNX → `mtcore.lifter_pose.LifterPose` (restricted to the output bones, 18 of 127 MHR joints) → `(bones, confidence, present)`. `set_avatar(rest_positions | None)`, `reset()` (on camera switch). |
| `vmc.py` | Own OSC encoder (`message`, `bundle`) and `VmcSender`. See VMC conventions. |
| `cli.py` | `mtservice` entry point and main loop; standalone or `--stdio`. `--dump-poses` / `--dump-landmarks` write JSONL for offline comparison; `--vmc off` disables sending. |

Default bundle: `data/m1/runs/m1-v10/live/` (gitignored, CMU-derived weights; never committed).

## Stdio protocol (`--stdio`)
JSON lines; stdout carries only events (MediaPipe and log output go to stderr). Authoritative list: `cli.py` docstring.
- Commands (stdin): `{"cmd":"avatar","rest_positions":{vrmBone:[x,y,z]} | null}` (null = bundle's default avatar, AvatarSample_D), `{"cmd":"camera","index":n}` (ignored for video sources), `{"cmd":"cameras"}`, `{"cmd":"stop"}`.
- Events (stdout): `ready` (model source, bones, source), `status` (~1/s: fps, detect_ms, model_ms, latency_ms capture → output, present fraction), `cameras`, `error`, `stopped` (reason).
- Closing stdin stops the runtime, so it never outlives its parent. Unknown or malformed commands produce an `error` event, not an exit.

## VMC conventions
- One OSC bundle per frame: `/VMC/Ext/OK 1`, `/VMC/Ext/T`, `/VMC/Ext/Root/Pos` (identity), one `/VMC/Ext/Bone/Pos` per output bone, then the extensions.
- Bone names: Unity HumanBodyBones (`UNITY_NAMES`): hips, spine, chest, upperChest, neck, head, shoulders, upper/lower arms, hands. Bones without a Unity name are skipped.
- Rotation sent as `(x, -y, -z, w)`: vspark's `rhylive_bone_mapper` applies the same self-inverse map, so vspark receives exactly the lifter's VRM normalized locals. Positions: the avatar's rest offsets from the nearest present parent (`rest_offsets`), mapped `(-x, y, z)`.
- Extensions: `/MT/Present 0|1`, `/MT/Conf <lifter joint> <0..1>` (confidence = exp(-log_scale)). Namespace `/MT/` is provisional (`PLAN.md` §7.2, `[OPEN]`); receivers ignore unknown addresses.
- No smoothing: output goes out as is at 10 FPS. Whether vspark smooths VMC input (the model was judged through glide) is an open question in the plan.

## Checks (2026-11-12, recorded seated-v1 session video)
- Bones within 0.12 deg of offline `mttrain.infer` on the same landmarks; ONNX vs torch 3e-6.
- Per frame: detection ~21 ms (CPU), features + ONNX + VRM conversion ~1.4 ms.
- Tests (`service/tests/`, `uv run pytest`): VMC encoding against vspark's mapping; streaming vs offline inference (skips without the local bundle and session).

## Extending
- **New frame source**: a class with `__iter__` yielding `(t seconds, BGR frame)` at the lifter's frame rate (10 FPS; the model's history offsets assume it) and `close()`; `CameraSource` also exposes `t0` for the latency stat. Wire it into `open_source` in `cli.py`. Keep the recorder's capture settings (mirroring happens in `detector.prepare`).
- **New output sink** (e.g. the deferred WebSocket): take `(t, bones, conf, present)` per frame like `VmcSender.send`, plus a rest-pose update hook if it sends positions; call it from the main loop next to `vmc.send`. New VMC extension addresses go under the same (provisional) namespace.
- **New bundle version**: `lifter.json` carries `"format": "motiontracker-lifter", "version": 1`. `LiveLifter` currently checks only the input layout (offsets, frame dim) against `mtcore.features`, not the version. When the bundle format changes (new inputs such as hands/face, other output bones), bump `version` in `mttrain.export` and branch on it in `LiveLifter`; feature changes go into `mtcore.features` so training, export and runtime stay identical.
- **New output bones**: come from the bundle (`output_bones`); add their Unity names to `vmc.UNITY_NAMES` and parents to `vmc.VRM_PARENT`.

## Open
- vspark behavior `learned_tracker` (planned, separate vspark branch). Checking the stream with vspark's existing `vmc_receiver` (plan step 3) is not recorded as done; only the unit test against its mapping exists.
- vspark-side smoothing, camera naming (OpenCV indices vs the browser's device list), camera exclusivity with MediaPipe Tracking, the OSC namespace name (plan open questions).
