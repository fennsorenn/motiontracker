# eval-harness

Status: implemented. Usage, system definitions, metric table and known limits: `eval/README.md` (authoritative). This file covers structure and how to extend it.

## Structure

uv project in `eval/`, package `mteval` (`src/mteval/`), CLI `mteval evaluate`. If sandboxed, uv may need `UV_CACHE_DIR` set.

- `quat.py` — quaternion helpers, including `slerp`.
- `session.py` — `Session` / `Segment` (optional `variant`) / `PoseTrack` loading (recorder session dirs or bare landmarks files). A session is named after its folder, not the `name` in `session.json`. `auto_still_segments`: windowed-average landmark speed, threshold 0.12 m/s (set from one static clip, rough).
- `metrics.py` — label-free and labeled metrics. All time-based so systems with different output rates are comparable. `xcorr_lag` linearly resamples midpoint-stamped speeds onto a common grid to stay unbiased across rates.
- `systems.py` — system registry.
  - Builtins `baseline` and `baseline-rendered`: run the baseline CLI (`baseline/`, needs `pnpm install`), with head calibration captured at the middle of the first `still` segment.
  - `name=path.jsonl`: precomputed `OutputFrame` JSONL (`{t, bones:{vrmBone:[x,y,z,w]}}`).
  - `name-rendered=path.jsonl` (`run_rendered`): runs a precomputed track through the baseline CLI's filter-only mode (vspark's One Euro filter, 60 Hz; `baseline.md`). The baseline's own solver output through it reproduces `baseline-rendered` exactly (end-to-end test).
  - `name-glide=path.jsonl`: upsamples a precomputed track to 60 Hz without smoothing, via `upsample.py`.
- `upsample.py` — `glide`: on each new pose, every bone slerps from what is currently shown to the new pose, finishing one render frame before the next pose is expected, then holds. Expected interval = mean of the last 10 input intervals (causal). Bones absent from the newest pose are absent from the output; reappearing bones snap. Tests: `tests/test_upsample.py`.
- `report.py` — `evaluate` → JSON; summary is the median per bone group (torso/head, shoulders, arms, hands, fingers); markdown rendering.
  - Pooled rest metrics are over all `still` segments together. Additionally `per_still_segment` per system and a top-level `still_segment_summary` break rest noise / twitches down per still segment (keyed by `variant` where present); the markdown adds a "Rest by still segment" table when a session has more than one still segment.
- `cli.py` — `mteval evaluate`. Reports default to `<repo>/reports/<session folder>/` regardless of cwd.
- `tests/test_metrics.py` — synthetic known-answer tests for every metric. `tests/test_end_to_end.py` — runs the real baseline on vspark's `hand_dropouts` clip.

## Extending

- Evaluating a learned model: write its output as `OutputFrame` JSONL and pass `--system name=path.jsonl` (raw), `name-glide=` (primary rendered comparison) or `name-rendered=`. This is the intended extension point; a builtin in `systems.py` is only worth it if the harness should run the model itself.
- New metric: add it to `metrics.py` as a time-based function, add a synthetic known-answer test, wire it into `report.py`, and add a row to the README metric table.
- Labeled metrics are ready but not reported; wire them into `report.py` once labeled sessions exist (M1).

## Caveats

- Compare systems at the same stage only: solver vs solver (`baseline`), rendered vs rendered (`baseline-rendered`). The rendered stage shows higher peak speeds and twitch counts from catch-up bursts after each 10 FPS pose, even though rest jitter and noise are lower.
- M1 go/no-go (owner, 2026-11-08, `PLAN.md` §6): the model replaces both vspark's solver and its One Euro filter, so the primary comparison is `name-glide` vs `baseline-rendered`; `name-rendered` (model through the filter) is secondary.
- Lag floor at 60 Hz: the lag metric stamps speeds at interval midpoints, so any causal 60 Hz display of 10 FPS input measures at least ~50 ms. Glide measures ~90 ms, about the same as `baseline-rendered` (70–95 ms); read lag differences between rendered systems against this floor.
- Lag is unreliable on short or low-motion clips; check `corr` in the JSON.

## Measurements so far (static test clip only)

- MediaPipe pose world landmarks jitter about 5 cm/s median at rest.
- Hands are the noisiest bones at rest: rest noise about 3.6 deg (solver), 2.1 deg (rendered).

## First real session (2026-11-01)

- Baseline hand rest noise depends strongly on rest posture: about 9.9 deg with arms below the frame vs about 1.8 deg with elbows in view. This motivated the `seated-v2` protocol and the per-still-segment breakdown; pooled rest metrics hide the difference.

## Not yet

Contact preservation, reprojection error (2D landmarks are already recorded). Success thresholds are `[OPEN]` (`PLAN.md` §6).
