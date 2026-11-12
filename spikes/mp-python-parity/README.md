# Spike: Python MediaPipe vs the recorder's tasks-vision (WASM) landmarks

Question (plan `dev-notes/plans/m2-vspark-runtime.md`, step 0): can a Python runtime reproduce the
landmarks m1 models were calibrated on (tasks-vision 0.10.17 WASM, VIDEO mode, mirrored 320x240,
10 FPS)? Answer (2026-11-12): yes, closely enough.

- `run_python.py`: mediapipe 0.10.18 (closest Python release to 0.10.17; no 0.10.17 wheel exists),
  same `.task` model, same options and frame selection as the recorder's stepped replay
  (`ts = k * 0.1 s + 1 ms`, latest frame at or before `ts`, mirrored, resized to 320x240), same
  output format as `landmarks.jsonl`.
  - 0.10.18's `HolisticLandmarker.detect_for_video` aborts the process on an empty output packet
    (e.g. no hand) and returns nothing at all when the face is missing. The script reads each output
    stream itself; the runtime must do the same.
- `compare.py`: presence, landmark differences and m1 bone differences between landmark sets.

Results on `sessions/2026-11-01T15-29-11-session` (video remuxed for seeking; local only):

| median / p95 | Python vs WASM replay | WASM replay vs live recording |
|---|---|---|
| wrists, image px | 1.7 / 10.0 | 2.5 / 15.5 |
| wrists, world cm | 1.2 / 4.3 | 1.5 / 6.2 |
| m1-v10 upper arms, deg | 1.9-2.3 / 8.2 | 2.7-3.2 / 10-12 |
| m1-v10 forearms, deg | 1.7-1.8 / 8.0 | 2.1-2.3 / 9-10.5 |
| m1-v10 head, deg | 0.6 / 2.0 | 0.9 / 2.6 |
| hand presence agreement | 96-98 % | 94-97 % |

- Two WASM replays are byte-identical: the WASM side is deterministic, so the Python gap is real.
- It is smaller than the gap between the same WASM detector on live frames and on the compressed
  recording, i.e. within normal capture variation.
- INTER_LINEAR and INTER_AREA resizes give identical landmarks (exact 2x downscale).

Run (from the repo root, CPU jobs under the guard):

    tools/guarded-run.sh -c 8-15 uv run --project spikes/mp-python-parity python spikes/mp-python-parity/run_python.py \
        <video> data/spikes/mp-depth/model/holistic_landmarker.task <out>/landmarks.jsonl
    tools/guarded-run.sh -c 8-15 uv run --project train python spikes/mp-python-parity/compare.py \
        data/m1/runs/m1-v10/best.pt <dirA> <dirB> [...]
