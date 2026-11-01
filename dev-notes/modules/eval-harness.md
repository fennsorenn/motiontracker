# eval-harness

Status: implemented. Usage, system definitions, metric table and known limits: `eval/README.md` (authoritative). This file covers structure and how to extend it.

## Structure

uv project in `eval/`, package `mteval` (`src/mteval/`), CLI `mteval evaluate`. If sandboxed, uv may need `UV_CACHE_DIR` set.

- `quat.py` — quaternion helpers.
- `session.py` — `Session` / `Segment` / `PoseTrack` loading (recorder session dirs or bare landmarks files). `auto_still_segments`: windowed-average landmark speed, threshold 0.12 m/s (set from one static clip, rough).
- `metrics.py` — label-free and labeled metrics. All time-based so systems with different output rates are comparable. `xcorr_lag` linearly resamples midpoint-stamped speeds onto a common grid to stay unbiased across rates.
- `systems.py` — system registry.
  - Builtins `baseline` and `baseline-rendered`: run the baseline CLI (`baseline/`, needs `pnpm install`), with head calibration captured at the middle of the first `still` segment.
  - `name=path.jsonl`: precomputed `OutputFrame` JSONL (`{t, bones:{vrmBone:[x,y,z,w]}}`).
- `report.py` — `evaluate` → JSON; summary is the median per bone group (torso/head, shoulders, arms, hands, fingers); markdown rendering.
- `cli.py` — `mteval evaluate`.
- `tests/test_metrics.py` — synthetic known-answer tests for every metric. `tests/test_end_to_end.py` — runs the real baseline on vspark's `hand_dropouts` clip.

## Extending

- Evaluating a learned model: write its output as `OutputFrame` JSONL and pass `--system name=path.jsonl`. This is the intended extension point; a builtin in `systems.py` is only worth it if the harness should run the model itself.
- New metric: add it to `metrics.py` as a time-based function, add a synthetic known-answer test, wire it into `report.py`, and add a row to the README metric table.
- Labeled metrics are ready but not reported; wire them into `report.py` once labeled sessions exist (M1).

## Caveats

- Compare systems at the same stage only: solver vs solver (`baseline`), rendered vs rendered (`baseline-rendered`). The rendered stage shows higher peak speeds and twitch counts from catch-up bursts after each 10 FPS pose, even though rest jitter and noise are lower. A learned model must be compared raw against `baseline`, or through the same render filter against `baseline-rendered`.
- Lag is unreliable on short or low-motion clips; check `corr` in the JSON.

## Measurements so far (static test clip only)

- MediaPipe pose world landmarks jitter about 5 cm/s median at rest.
- Hands are the noisiest bones at rest: rest noise about 3.6 deg (solver), 2.1 deg (rendered).

Real-session baseline numbers are still pending (next step: record sessions with the owner).

## Not yet

Contact preservation, reprojection error (2D landmarks are already recorded). Success thresholds are `[OPEN]` (`PLAN.md` §6).
