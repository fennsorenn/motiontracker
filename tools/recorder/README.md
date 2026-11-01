# Session recorder

A static browser page that records a test session: webcam video plus MediaPipe Holistic landmarks,
captured the way vspark does it. The evaluation harness (`eval/`) runs systems over these sessions.

## Run

```sh
python3 -m http.server 8000 -d tools/recorder
```

Open http://localhost:8000 in Chrome. The camera only works on `localhost` or HTTPS. Pick a
camera, video size and protocol, then press **Start recording**.

- **seated-v1** (about 70 s) prompts you through still, slow, fast, hand, head and hand-dropout
  segments. It stops by itself. The labels become `segments` in `session.json`.
- **seated-v2** (about 71 s) is the same, except the two still segments cover the two rest
  postures: arms below the camera view first, hands resting visibly on the desk last. They
  carry a `variant` tag in `session.json`. Prefer it for new recordings.
- **free** has no prompts. Press <kbd>S</kbd> to start and end a "still" segment. Press
  **Stop and save** when done.

On save, Chrome asks for a folder once. Choose the repo's `sessions/` folder; it is gitignored.
Each session goes into its own subfolder. Browsers without folder access download the files
with a name prefix instead; move them into `sessions/<name>/` without the prefix.

To replay an existing video through the same pipeline (landmarks only, timestamps = video time),
put it next to `index.html` and open `index.html?video=<file>`.

## What a session contains

| File | Content |
|---|---|
| `session.json` | Protocol, segments (`label`, `t0`, `t1` in ms), camera settings, detector version and model hash. |
| `landmarks.jsonl` | One frame per line: `t` (ms) plus `pose`, `face`, `leftHand`, `rightHand` exactly as vspark sends them, and `pose2d`, `leftHand2d`, `rightHand2d` (normalized image coordinates, for later reprojection metrics). |
| `video.webm` | The unmirrored camera stream (VP9/VP8, 8 Mbit/s). |

## Fidelity to vspark

Reproduced: tasks-vision 0.10.17, the same Holistic model (pinned to version 1, byte-identical
to vspark's `latest`), VIDEO running mode with the same confidence options, mirroring, 320×240
input, the 10 FPS throttle, and the same landmark selection per stream.

Differences:
- vspark asks the camera for 320×240 and processes whatever size it delivers. The recorder
  captures at the chosen size and draws a 320×240 copy for detection. MediaPipe resizes its
  input internally, so results should be close but are not guaranteed identical.
- Detection runs on the page's main thread, not in a worker. That changes speed, not output.
- Coordinates are rounded to 5 decimals.
- The video comes from the browser's MediaRecorder: compressed, with a variable frame rate.
  Landmark times count from the recorder's start event, so expect about one frame of
  misalignment between video and landmarks.

## Test

`test/run_headless.sh` builds a short video from MediaPipe's own test image of a person, replays
it through the page in headless Chrome and saves the session to a temp folder. It never uses
the webcam. The `?test` URL flag keeps the result in the page instead of saving files.
