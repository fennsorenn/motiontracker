# Plan: live runtime for vspark (camera → MediaPipe → m1-v10 → VMC), loaded lazily by a vspark behavior

> Branch: `feature/m2-vspark-runtime` (this repo); vspark `feature/learned-tracker` (pushed) · Status: steps 0-4 done (2026-11-12); step 5 (owner live test) open
> This plan is the seed context for a worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

Run the M1 lifter (m1-v10) live inside vspark so the owner can use and judge it in the real app.
A Python runtime in this repo owns the camera, runs MediaPipe Holistic and the model, and sends VRM
bone rotations as VMC. vspark gets a native-feeling behavior that starts that runtime lazily from
this repo (not bundled) and feeds its output into vspark's graph. This is the first slice of M2
(`PLAN.md` §7, decisions in §7.4).

## Decisions

- One runtime owns camera + MediaPipe + model (feature taps later need the MediaPipe models). `[USER]` (2026-11-12)
- Python POC runtime. `[USER]` (2026-11-12)
- vspark behavior kind `learned_tracker`, label "Learned Tracking". `[USER]` (2026-11-12)
- Runtime not bundled into vspark: started through `uv` from a local checkout of this repo (config
  path); from the public git URL (`uvx --from git+...#subdirectory=service`) later. `[USER]` (2026-11-12)
- Output VMC over localhost UDP; confidence/presence under an own OSC namespace. `[USER]` (2026-11-12)
- Control over the child process's stdin/stdout as JSON lines; WebSocket deferred. `[USER]` (2026-11-12)
- vspark passes the avatar's rest bone positions at start; VRoid rest as default. `[USER]` (2026-11-12)
- The model drives every bone it predicts (head included); vspark's mixer combines sources. `[USER]` (2026-11-12)
- Model m1-v10. `[USER]` (2026-11-12)
- Runtime dependencies: mediapipe, onnxruntime, numpy, opencv, an OSC encoder. No torch or pymomentum
  at runtime: the model is an ONNX export and MHR's rest skeleton an exported table. `[CLAUDE]`
- Capture settings as the recorder (mirrored, 320x240, 10 FPS, VIDEO mode) so inputs match what v10
  was calibrated on. `[CLAUDE]`
- `features.py` (numpy only) moves from `train` to `mtcore` so runtime and training share one
  implementation; a streaming form of `causal_scale` and `history_indices` for one frame at a time. `[CLAUDE]`

## Constraints

- Strictly causal, low latency; model cost < 2 ms/frame on CPU (go/no-go criterion 6).
- Weights stay in gitignored `data/`; never commit CMU-derived data. The repo is public since 2026-11-12.
- Check `docs/LICENSES-REGISTER.md` and add every new dependency (mediapipe Python, onnxruntime,
  opencv-python, OSC library) before use; permissive only.
- vspark: separate branch; its working tree has an untracked `ai-assistant-issues.md` on
  `feature/retire-macros`: ask the owner before branching there.
- CPU-heavy checks through `tools/guarded-run.sh -c 8-15`.

## Files in scope

- `core/src/mtcore/features.py` (moved from `train/src/mttrain/features.py`), train imports updated.
- `train/src/mttrain/export.py` (new): ONNX export of a checkpoint, MHR rest table export.
- `service/` (new package `mtservice`): sources, MediaPipe, model runner, MHR → VRM, VMC sender, stdio control, CLI.
- `docs/LICENSES-REGISTER.md`, `dev-notes/`.
- vspark: `packages/backend/src/behaviors/learned_tracker/` (new), registry, frontend spots that
  special-case `mediapipe_tracker` (`icons.tsx`, `MediaInputWindow.tsx`, `SceneGraph.tsx`, `Viewport.tsx`).

## Out of scope

- WebSocket output, browser viewer, remote media input.
- Face blendshapes / hands / fingers from the runtime (possible next slice for head mixing).
- Feature taps, native core, model changes (v11), publishing weights.

## Approach

0. **Parity check (decides the approach).** Can Python MediaPipe reproduce the recorder's landmarks?
   - Python Tasks API Holistic Landmarker available? Pin the version closest to tasks-vision 0.10.17
     and the same `.task` model (MD5 in the register).
   - Feed the recorded session's video (`sessions/2026-11-01T15-29-11-session/video.webm`) through it
     at the recorder's settings; compare landmarks with `landmarks.jsonl` (per-landmark 2D/3D error,
     visibility, dropout timing) and run v10 on both (bone angle difference).
   - If it diverges materially: stop and report to the owner before building on it.
1. **Exports.** ONNX of the v10 checkpoint (parity vs torch < 1e-4); MHR rest table (joint names,
   parents, rest state) as JSON so the runtime needs no pymomentum. `MhrToVrm` takes the avatar's
   rest positions as given (vspark) or the VRoid default.
2. **Runtime (`service/`).**
   - Sources: camera (OpenCV), video file (repeatable tests, frame timestamps from the file).
   - MediaPipe Holistic in VIDEO mode → per-frame features (streaming) → ring buffer → ONNX → VRM locals.
   - VMC sender: `/VMC/Ext/Bone/Pos` per bone (Unity left-handed conversion), `/VMC/Ext/Root/Pos`,
     `/VMC/Ext/OK`, `/VMC/Ext/T`; own namespace for per-joint confidence and presence (name `[OPEN]`).
   - Stdio control: JSON lines in (start, stop, camera, avatar rest, target port), out (ready, status,
     errors, camera list, timing).
   - CLI for standalone use (`mtservice --camera 0 --vmc 127.0.0.1:39539`).
3. **Runtime checks.** Replay the session video through the runtime and compare its bones with
   offline `mttrain.infer` on the same landmarks; per-frame latency (capture → send); CPU use.
   Confirm the stream with vspark's existing `vmc_receiver` before writing the new behavior.
4. **vspark behavior `learned_tracker`** (vspark branch).
   - Config: runtime path (local checkout), camera, port; start/stop the child process with the
     behavior, restart on crash with backoff, surface status and errors.
   - Receives the VMC stream (reuse the `vmc_receiver` parsing) into a graph template like
     `mediapipe_tracker`'s outputs, so the mixer and calibration nodes apply.
   - Sends the avatar's rest bone positions (`loadVrmSkeleton`) at start.
   - Frontend: add the kind where `mediapipe_tracker` is special-cased, or a shared "is tracking
     behavior" check.
5. **Owner test live**, side by side with MediaPipe Tracking.

## Open questions

- Does vspark smooth VMC input (One Euro at render)? v10 was judged through glide (no smoothing);
  the behavior may need its smoothing off or a glide equivalent.
- Camera naming: OpenCV indices vs the browser's device list; the runtime lists cameras with names.
- Camera exclusivity: MediaPipe Tracking and Learned Tracking can't share one webcam on Linux.
- OSC namespace name for the extensions (`PLAN.md` §7.2).

## Acceptance / verification

- Parity: Python vs recorded landmarks close enough that v10's bones differ by well under the
  model's own error (target: median < 2 deg on arms); reported with numbers.
- Runtime bones match offline inference on the same landmarks (< 0.1 deg).
- Model cost < 2 ms/frame; end-to-end added latency reported.
- vspark: enabling "Learned Tracking" starts the runtime, the avatar moves, disabling stops it; no
  bundled Python in vspark's build.

## Output

Commits on `feature/m2-vspark-runtime` (and the vspark branch); report to the owner before merging into `dev`.

## Progress (2026-11-12)

- Steps 0-3 done in this repo (parity spike: go; mtcore features + lifter_pose; mttrain.export; service/).
- Step 4 done in vspark `feature/learned-tracker` (pushed, not merged): `learned_tracker` as a third
  VmcManager kind, LearnedRuntime process manager, cameras route, properties panel, i18n, help, tests;
  `/MT/Present` drives tracking on/off (plain VMC Bone/Pos never turned vspark's tracking on).
  End-to-end with the real runtime on a video: poses arrive through vspark's mapping unchanged.
- Runtime additions: `avatar` command takes the avatar's `vrm_path` (read with mtcore.vrm), `--avatar-vrm`.
- Open: step 5 (owner live test); vspark's One Euro render filter stays on for now (owner).
