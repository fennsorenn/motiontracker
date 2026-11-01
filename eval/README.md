# Evaluation harness (`mteval`)

Runs tracking systems over a recorded session and reports jitter, twitch, dropout and lag
metrics. Sessions come from the recorder in `tools/recorder/`. The baseline is vspark's solver
in `baseline/`.

## Usage

```sh
cd eval
uv run mteval evaluate ../sessions/<session>            # baseline and baseline-rendered
uv run mteval evaluate ../sessions/<session> --system baseline --system mymodel=poses.jsonl
uv run pytest
```

The report goes to `reports/<session>/report.md` and `report.json`, or to `--out`.

- **Systems:** `baseline` is vspark's solver output, one pose per detector frame.
  `baseline-rendered` adds vspark's frontend One Euro filter at 60 Hz, which is what the
  avatar actually shows. Any other system can be evaluated from a precomputed file of
  `{t, bones: {vrmBone: [x, y, z, w]}}` lines.
- **Head calibration:** for the baseline, the head calibration is captured in the middle of
  the first "still" segment, the way a vspark user would press the button. Use
  `--no-calibration` to skip it.
- **Sessions without segments:** a bare landmarks file or a free recording has no segments.
  `--auto-still` then guesses still stretches from the landmarks. This is rough, so prefer
  protocol recordings.

The baseline must be installed (`pnpm install` in `baseline/`). If `uv` can't write to its
cache, set `UV_CACHE_DIR`.

## Metrics

All metrics are time-based, so systems with different output rates are comparable. The
report gives the median over the bones in each group: torso and head, shoulders, arms, hands,
fingers.

| Metric | Definition | Needs |
|---|---|---|
| Rest jitter | RMS angular speed (deg/s) during still segments. | still segments |
| Rest noise | RMS angle (deg) between each pose and the mean pose over ±250 ms, during still segments. | still segments |
| Twitches per minute | Runs of angular speed above 180 deg/s during still segments. | still segments |
| Speed p50/p90/p99 | Angular-speed percentiles, overall and per segment label. | – |
| Hand dropout peak | Peak angular speed of a side's arm, hand and finger bones within 500 ms after that hand appears or disappears in the input. Averaged over transitions. | – |
| Lag vs detector | Delay (ms) at which the bone's angular speed best correlates with the detector's: upper-arm and forearm directions from the pose landmarks, and head direction for the neck. Counted only when the correlation is at least 0.3. This is what the solver and smoothing add on top of MediaPipe, not camera or display latency. | motion |
| Presence | Fraction of output frames in which the bone exists. | – |

Labeled metrics are implemented and tested on synthetic signals. They are not in the report
yet, because no labeled sessions exist. They are rotation error (mean, median and p95
geodesic error), onset lag, and peak-velocity ratio (predicted over true peak speed per
movement).

### Comparing stages

Compare systems at the same stage, never across stages. After rendering, a 10 FPS stream
shows each pose change as a quick catch-up over a few 16 ms frames. That raises peak speeds
and twitch counts for the rendered stage, even though its rest jitter and noise are lower.
The fair comparison for a learned model is its own output against `baseline`, or its output
through the same render filter against `baseline-rendered`.

### Known limits

- The still-detection threshold was set from one static test clip.
- Lag is unreliable on short clips or clips with little motion. Check the `corr` value in the
  JSON.
- Contact preservation and reprojection error (PLAN.md §6) are not implemented yet. The
  recorder already saves the 2D landmarks reprojection will need.
- Success thresholds are still open (PLAN.md §6).
