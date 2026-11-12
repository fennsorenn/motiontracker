# mtservice: live runtime

Camera → MediaPipe Holistic → M1 lifter (ONNX) → VRM bone rotations as VMC over UDP. Started and
controlled over stdio by vspark's "Learned Tracking" behavior, or run on its own. Plan:
`dev-notes/plans/m2-vspark-runtime.md`; decisions: `PLAN.md` §7.4.

## Setup

1. Export a model bundle (needs the `train` environment and a checkpoint):

       cd train && uv run python -m mttrain.export ../data/m1/runs/m1-v10/best.pt --check-session ../sessions/<session>

   This writes `data/m1/runs/m1-v10/live/` (`model.onnx`, `lifter.json`), the runtime's default bundle.
   The weights are trained on CMU-derived data and stay local (license register).
2. `cd service && uv sync`. The Holistic model (`holistic_landmarker.task`, float16 v1, pinned by MD5)
   is downloaded once into `~/.cache/motiontracker/` unless `--task` points at a copy.

## Run

    uv run mtservice --camera 0                 # VMC -> 127.0.0.1:39539 (vspark's VMC Receiver default)
    uv run mtservice --list-cameras
    uv run mtservice --video <file> --realtime  # a recording instead of the camera
    uv run mtservice --video <file> --vmc off --dump-poses poses.jsonl --dump-landmarks landmarks.jsonl

`--stdio` turns on the control protocol for a parent process (JSON lines; full list in
`src/mtservice/cli.py`): commands `avatar` (rest bone positions of the avatar being driven, or null
for the default VRoid rest), `camera`, `cameras`, `stop`; events `ready`, `status` (about once a
second: fps, detection and model time, capture-to-output latency, pose presence), `cameras`,
`error`, `stopped`. Closing stdin stops the runtime, so it never outlives its parent.

## What it reproduces

Everything the lifter was calibrated on (tools/recorder): the same Holistic model with the same
options, VIDEO mode, mirrored 320x240 input, 10 FPS. Python MediaPipe 0.10.18 gives landmarks close
to tasks-vision 0.10.17 (closer than live vs recorded frames; `spikes/mp-python-parity`). 0.10.18's
Holistic Python wrapper aborts on missing hands and drops the pose without a face, so `detector.py`
reads the output streams itself.

## Output

- `/VMC/Ext/Bone/Pos` for hips, spine, chest, upperChest, neck, head, shoulders, arms and hands
  (Unity bone names). Rotations in VMC/Unity convention: vspark maps them back with (x, -y, -z, w)
  to exactly the lifter's output. Positions: the avatar's rest offsets.
- `/VMC/Ext/Root/Pos` (identity), `/VMC/Ext/OK`, `/VMC/Ext/T`.
- Extensions: `/MT/Present` (0/1), `/MT/Conf <joint> <0..1>` per lifter joint. Namespace provisional.
- No smoothing: the lifter's output is sent as is at 10 FPS.

## Checks (2026-11-12, recorded session video)

- Landmarks identical to the parity spike; bones within 0.12 deg of offline torch inference on the same
  landmarks (ONNX vs torch: 3e-6).
- Per frame: detection about 21 ms (CPU), features + ONNX + VRM conversion about 1.4 ms.
- Tests: `uv run pytest` (VMC encoding against vspark's mapping; streaming vs offline inference when
  the local bundle and session exist).
