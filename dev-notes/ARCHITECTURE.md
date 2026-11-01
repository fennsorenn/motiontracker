# Architecture

Learned live motion tracking for VRM 1.0 avatars: webcam → MediaPipe detectors → small causal learned heads + merger → bone rotations, root translation, blendshapes, gaze → VMC/OSC. Full context, decisions and open questions: `PLAN.md` (decision tags there are authoritative; `[CLAUDE]` items are unconfirmed).

No learned-model or service code exists yet. Current milestone: M0 (baseline + evaluation harness, `PLAN.md` §10).

## Modules

| Module | Status | Summary |
|---|---|---|
| `baseline` | implemented (frozen) | Frozen snapshot of the owner's current algorithmic MediaPipe → VRM bone solver (vspark), for comparison. Details: `modules/baseline.md`. |
| `eval-harness` | planned | Metrics and test sets for comparing baseline vs learned models (`PLAN.md` §6). |

### baseline (implemented, frozen)
Location: `baseline/` (own pnpm/TypeScript package). Byte-identical copies of vspark's solver nodes under `baseline/vendor/`, driven by a minimal standalone pipeline (`src/pipeline.ts`) instead of vspark's graph engine. CLI: MediaPipe landmark frames JSONL in → `OutputFrame` JSONL out (`{t, bones:{vrmBone:[x,y,z,w]}}`). Verified bit-exact against vspark's real engine. Usage, provenance and fidelity quirks: `baseline/README.md`.

- Frozen: the reference the learned pipeline must beat. Never edit `vendor/` to improve it.
- Notable reproduced quirk: vspark holds the last hand/face landmarks when a stream drops out (on by default, `holdMissing`).

### eval-harness (planned)
Consumes per-frame `OutputFrame` JSONL (baseline CLI first, learned models later) and computes `PLAN.md` §6 metrics: rotation error, onset lag, peak-velocity ratio, jitter at rest, contact preservation, reprojection error. Success thresholds are `[OPEN]`. Design not started.

## Key decisions
- Build the evaluation harness and measure the baseline before training any model (`PLAN.md` §6).
- Baseline source is the vspark branch including uncommitted WIP, judged more accurate than its last commit (`[USER]`, 2026-11-01).
- Baseline extraction method: frozen copy (`[USER]`, 2026-11-01, `PLAN.md` §1.4).

## Licensing
`docs/LICENSES-REGISTER.md` tracks source, license and usage of every asset, dataset, model and recording. The vendored vspark code has no declared license; the owner must relicense it before any public release.
