# Plan: m1-v10, MediaPipe-like head response without noise, plus more head motion

> Branch: `feature/m1-head-obs` · Status: draft
> This plan is the seed context for a worker. It is a starting point, not an
> airtight spec — the worker is interactive and may ask to refine it.

## Goal

Get head movement relative to the torso back into the pose model's output. m1-v8 follows the
head-vs-chest rotation at only ~0.1-0.35 of its true size; m1-v9 (gain + noise) dropped to ~0.

Diagnosis (2026-11-11, head test clip `data/m1/renders/head1/seq00`, MediaPipe VIDEO mode vs truth):

| MediaPipe vs truth | turn | nod | tilt |
|---|---|---|---|
| 2D head cues, slope / corr | 0.63 / 0.96 | 0.44 / 0.82 | 0.35 / 0.57 |
| 3D head vs chest, slope / corr | 0.05 / 0.38 | 0.04 / 0.23 | 0.07 / 0.55 |

The 2D signal is strong: a steady gain with high correlation, which a model can invert if training
shows it the same gain. The 3D head is locked to the torso. v8 trained on synthetic points that follow
the head fully in 2D and 3D, so it leaned on the 3D head (absent live) and read the 2D cues at full
strength. v9 added 8 deg of 2D noise (taught the model to ignore 2D) and turned the face about the
neck joint (moved it ~21 px). The low correlations quoted earlier (~0.5) came from the CMU test sets,
which have little head-vs-chest motion; they are not a ceiling.

## Decisions

- No added head noise in either space. `[USER]` (2026-11-11)
- More training samples with real head motion. `[USER]` (2026-11-11)
- Head only: the seated tremor and wrist-error changes (proposed for v10 earlier) stay out, so the
  run isolates the head change. `[CLAUDE]`
- Hand lock off (gain 1, noise 0, as v8): never validated, part of v9's regression. Code stays. `[CLAUDE]`
- 3D head dropout (`HEAD3D_DROP` 0.3) kept: harmless once the 3D head is locked, and keeps
  `--no-head3d` available for comparison. `[CLAUDE]`

## Constraints

- Numpy (`mpstyle`, `synth`, `seated`) and GPU (`gpugen/*`) paths stay mirrors; tests compare them.
- Arms must not regress on the four rendered sets (v1, fov1, rest1, contact1).
- Training as v8/v9: m1 lifter, `MT_FITS=fits-v3`, `CUDA_VISIBLE_DEVICES=1` (RTX 4060; the 3090 is
  taken), ~40 min. CPU jobs through `tools/guarded-run.sh -c 8-15`, one heavy job at a time.
- CMU-derived data stays in gitignored `data/`.

## Files in scope

- `train/src/mttrain/mpstyle.py`, `gpugen/geom.py` — head response: per-axis gain, pivot, 3D lock
  with nod offset; constants and Style sampling.
- `train/src/mttrain/seated.py`, `gpugen/motion.py` — head move generator (more head motion).
- `train/src/mttrain/synth.py`, `gpugen/generator.py` — wiring only if signatures change.
- `train/src/mttrain/headcal.py` — calibrate against the head clips (per-axis cues, 3D offset).
- `train/src/mttrain/render.py` — only to re-render the second head clip (seq01 hung on replay).
- `train/tests/test_synth.py`, `test_gpugen.py` — updated / new tests.

## Out of scope

- Seated tremor, wrist error model, hand lock calibration (later run).
- Face-mesh head (face model, PLAN.md §3.11) and the merger.
- Undoing MediaPipe's own smoothing.

## Approach

1. **Second calibration clip.** Re-render `head1` seq01 (other body) with `render prepare --head`
   and replay in VIDEO mode; if it hangs again, render a new seed. Two bodies instead of one.
2. **Head response (`head_lock`), per space:**
   - Decompose the head-vs-chest rotation (HeadRig delta) in the landmark chest frame and scale the
     rotation-vector components per axis: turn (about chest up), nod (about the ear line), tilt
     (about chest forward).
   - Pivot: the ear midpoint of the landmarks being edited (not `c_neck`), so the face doesn't move.
   - Image gains: chosen so the synthetic *cue* slopes on the head clips match MediaPipe's (target
     ~0.63 / 0.44 / 0.35; synthetic placement already changes the cue slope, v8 measured 0.96 / 1.11
     / 0.96 at gain 1, so the rotation gains will be roughly 0.65 / 0.4 / 0.36). Calibrate with
     `headcal` extended to `--sets head1`.
   - World: near-locked (gain ~0.05, calibrated the same way) plus a fixed nod rotation of the 3D
     face so its constant nod matches MediaPipe's (~+7..10 deg vs synthetic ~-13; measure on both
     clips).
   - Noise: 0 (`HEAD_NOISE_DEG` 0; the code path stays).
   - Per-sequence spread: per-axis gains log-normal ~0.2 (was 0.4), clipped to [0.02, 1]; the
     renders' faces are untextured and real faces may respond somewhat more.
3. **More head motion (`seated` + `gpugen/motion`):** a head move generator for a fraction of seated
   sequences (proposed 40 %): min-jerk moves to random targets (turn ±70 deg, nod −30..+35, tilt
   ±30; each axis independently zero with p 0.4 so isolated turns/nods/tilts occur), move 0.3-1.2 s,
   hold 0.2-2 s, split over neck and head with `head_choreography`'s shares, added to CMU's head and
   clamped to the limits. Random targets, not the test clip's `HEAD_POSES` list. Existing
   `HEAD_LIFE` stays for the other seated sequences. Also applied to the contact choreography pool:
   no (it is pre-generated; leave it).
4. **Tests:** per-axis gain (a pure turn scaled by the turn gain only, etc.), pivot (ear midpoint
   fixed), numpy vs torch for the new head lock and head moves, head move statistics (range,
   clamped), existing 21 tests pass.
5. **Check the data before training:** `headcal --sets head1` synthetic vs MediaPipe slopes per axis
   in both spaces; head-vs-chest range of a sample of training sequences.
6. **Train m1-v10.** Check at ~60 s and ~600 s (loss sane, no NaN, GPU memory).
7. **Evaluate** m1-v10 vs m1-v8 (and v9 for reference).

## Acceptance / verification

- Head clips (head1 seq00, seq01): head-vs-chest slope/corr per axis, v10 vs v8. Target: turn and
  nod slope well above v8's (≥ 0.6), tilt improved; no added jitter (roughness vs truth).
- Four rendered CMU sets: arm, forearm, hand errors within ~1 deg/cm of v8; head error not worse.
- Recorded session (`sessions/2026-11-01T15-29-11-session`): head-vs-chest range and roughness vs
  v8 and the baseline, plus a comparison video for the owner.

## Output

Commits on `feature/m1-head-obs`; report results to the owner before merging into `dev`.
