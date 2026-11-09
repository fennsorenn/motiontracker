# recorder

Status: implemented. Usage, session file contents and fidelity to vspark: `tools/recorder/README.md` (authoritative). This file covers structure and how to extend it.

## Structure

Static page, no build step (serve `tools/recorder/` over localhost).

- `index.html` — UI shell.
- `recorder.js`
  - Capture loop mirroring vspark's CameraCapture (`_tick` / `_drawMirror` / `_dispatch`): mirrored 320x240 copy, 10 FPS throttle, HolisticLandmarker in VIDEO mode on the main thread.
  - Protocol runner: steps through a protocol's prompts and records each step as a timed segment.
  - Save: File System Access API (one folder pick, one subfolder per session), download-with-prefix fallback for other browsers.
  - `?video=<file>` replay mode: feeds a video file through the same pipeline; timestamps are video time.
  - `?test`: keeps the result in `window.__mtResult` instead of saving files (used by the headless test).
- `protocols.js` — `PROTOCOLS` map (`seated-v1`, `seated-v2`, `free`).
  - `seated-v2` (~83 s): same as `seated-v1`, but the two `still` segments carry a `variant`: first `arms-below-frame` (10 s), last `hands-on-desk` (10 s); plus two `contact` steps (2026-11-09, criterion 7b): `palms-together` and `hand-to-face`, 6 s each. Arms resting below the frame is a normal streaming posture that must look natural (owner, `PLAN.md` §3.3, 2026-11-01). `seated-v1` is unchanged for comparability with earlier sessions.
- `test/run_headless.sh` + `test/drive_headless.mjs` — headless Chrome driven over CDP, replaying a short video built from MediaPipe's Apache-licensed test image. Never touches the webcam. `drive_headless.mjs` takes an optional timeout argument; `mttrain.render replay` uses it to run rendered videos through the exact live MediaPipe build (`modules/train.md`).

## Session format

Directory per session, stored in gitignored `sessions/`:
- `session.json` — `format: "motiontracker-session"`, version 1; protocol, segments (`label`, optional `variant`, `t0`, `t1` ms), camera settings, detector version and model hash.
- `landmarks.jsonl` — per-frame vspark-format landmarks (what the baseline CLI consumes) plus extras `pose2d`, `leftHand2d`, `rightHand2d` (normalized image coords, kept for future reprojection metrics).
- `video.webm` — MediaRecorder output, ~1 frame misalignment with landmarks.

Changing the format: bump the version in `session.json` and update `eval/src/mteval/session.py` to load it.

## Extending

- New protocol: add an entry to `PROTOCOLS` in `protocols.js` with `leadIn` and `steps: [{label, duration, prompt, variant?}]`. Step labels become session segments; `recorder.js` copies an optional `variant` into the segment. Don't change an existing protocol's steps; add a new version (`-v2`, ...) so old sessions stay comparable.
- Segment labels are meaningful to the harness: `still` is what it uses for rest metrics (jitter, noise, twitches); `variant` distinguishes rest postures and gets its own row in the report's per-still-segment breakdown. `contact` segments feed the criterion 7b contact metric. Other labels get per-label speed percentiles. Reuse existing labels where the meaning matches.
- Keep the capture path faithful to vspark (see README "Fidelity"); any deliberate deviation should be listed there.
