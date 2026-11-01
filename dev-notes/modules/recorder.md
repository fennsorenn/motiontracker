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
- `protocols.js` — `PROTOCOLS` map (`seated-v1`, `free`).
- `test/run_headless.sh` + `test/drive_headless.mjs` — headless Chrome driven over CDP, replaying a short video built from MediaPipe's Apache-licensed test image. Never touches the webcam.

## Session format

Directory per session, stored in gitignored `sessions/`:
- `session.json` — `format: "motiontracker-session"`, version 1; protocol, segments (`label`, `t0`, `t1` ms), camera settings, detector version and model hash.
- `landmarks.jsonl` — per-frame vspark-format landmarks (what the baseline CLI consumes) plus extras `pose2d`, `leftHand2d`, `rightHand2d` (normalized image coords, kept for future reprojection metrics).
- `video.webm` — MediaRecorder output, ~1 frame misalignment with landmarks.

Changing the format: bump the version in `session.json` and update `eval/src/mteval/session.py` to load it.

## Extending

- New protocol: add an entry to `PROTOCOLS` in `protocols.js` with `leadIn` and `steps: [{label, duration, prompt}]`. Step labels become session segments.
- Segment labels are meaningful to the harness: `still` is what it uses for rest metrics (jitter, noise, twitches). Other labels get per-label speed percentiles. Reuse existing labels where the meaning matches.
- Keep the capture path faithful to vspark (see README "Fidelity"); any deliberate deviation should be listed there.
