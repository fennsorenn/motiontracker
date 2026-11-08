# Architecture

Learned live motion tracking for VRM 1.0 avatars: webcam → MediaPipe detectors → small causal learned heads + merger → bone rotations, root translation, blendshapes, gaze → VMC/OSC. Full context, decisions and open questions: `PLAN.md` (decision tags there are authoritative; `[CLAUDE]` items are unconfirmed).

No service code exists yet. Current milestone: **M1, the body lifter in MHR space** (WIP, plan: `plans/m1-body-lifter.md`): pipeline, GPU training-data generator, render stage and criterion 7/8 checks done, latest model `m1-v4`; the go/no-go still needs seated-v2 sessions. Done: M0 tooling plus the first measured session (seated-v1), and the M3 MHR ↔ VRM 1.0 spike (result: go for MHR as canonical skeleton). The face-cost check moves to M4.

## Modules

| Module | Status | Summary |
|---|---|---|
| `baseline` | implemented (frozen) | Frozen snapshot of the owner's current algorithmic MediaPipe → VRM bone solver (vspark), for comparison. Details: `modules/baseline.md`. |
| `eval-harness` | implemented | Python (`eval/`, package `mteval`): runs systems over recorded sessions and computes metrics (`PLAN.md` §6). Details: `modules/eval-harness.md`. |
| `recorder` | implemented | Static browser page (`tools/recorder/`) that records test sessions: raw video + MediaPipe landmarks + protocol segments. Details: `modules/recorder.md`. |
| `mhr-vrm-spike` | implemented (M3, done) | Spike (`spikes/m3-mhr-vrm/`): MHR → VRM 1.0 conversion, CMU → MHR fitting, round-trip eval. Reusable code now in `core/`. Details: `modules/mhr-vrm-spike.md`. |
| `m1-body-lifter` | WIP (M1) | First learned model: MediaPipe pose landmarks → small causal model → MHR upper-body rotations → VRM 1.0. Go/no-go for the learned approach. Plan: `plans/m1-body-lifter.md`. |
| `mtcore` | implemented | Shared Python package (`core/`): MHR loading/FK, CMU reader + CMU → MHR fit, MHR → VRM conversion, MediaPipe landmarks from MHR. Details: `modules/mtcore.md`. |
| `train` | implemented | uv project (`train/`, package `mttrain`, CUDA PyTorch): CMU fits, seated synthesis, synthetic landmarks + noise, batched GPU data generator, model, training, inference on sessions, limit check. Details: `modules/train.md`. |
| `retargeting` | planned (M4) | Learned layer mapping the performer's pose onto the avatar's MHR identity, then to VRM 1.0 (`PLAN.md` §3.10a). |

### baseline (implemented, frozen)
Location: `baseline/` (own pnpm/TypeScript package). Byte-identical copies of vspark's solver nodes under `baseline/vendor/`, driven by a minimal standalone pipeline (`src/pipeline.ts`) instead of vspark's graph engine. CLI: MediaPipe landmark frames JSONL in → `OutputFrame` JSONL out (`{t, bones:{vrmBone:[x,y,z,w]}}`). Verified bit-exact against vspark's real engine. Usage, provenance and fidelity quirks: `baseline/README.md`.

- Frozen: the reference the learned pipeline must beat. Never edit `vendor/` to improve it.
- Notable reproduced quirk: vspark holds the last hand/face landmarks when a stream drops out (on by default, `holdMissing`).
- Filter-only mode (`--poses poses.jsonl --stage rendered`, implemented): skips the solver and runs any pose track through `renderStage` (vspark's frontend One Euro filter, 60 Hz), exactly like the `baseline-rendered` stage. No change to `vendor/`.

### recorder (implemented)
Location: `tools/recorder/`. Reproduces vspark's CameraCapture path (mirrored, 320x240, 10 FPS, MediaPipe tasks-vision 0.10.17 HolisticLandmarker, VIDEO mode). Writes one session directory (`session.json` + `landmarks.jsonl` + `video.webm`) into gitignored `sessions/`. Can also replay a video file instead of the webcam. Usage and fidelity notes: `tools/recorder/README.md`.

### eval-harness (implemented)
Location: `eval/` (uv project, package `mteval`, CLI `mteval evaluate`). Runs systems over a session (baseline via its CLI at solver and rendered stages; any other system from a precomputed `OutputFrame` JSONL) and writes a JSON + markdown report. All metrics are time-based.
- Label-free metrics on real recordings: rest jitter/noise, twitch rate, speed percentiles, hand-dropout peaks, lag vs detector, presence, hand position on the performer's skeleton (criterion 7).
- Labeled metrics (rotation error, onset lag, peak-velocity ratio): tested on synthetic signals only, not in the report until labeled data exists (M1).
- Compare systems at the same stage only (solver vs solver, rendered vs rendered). A learned model counts as rendered when shown through glide or the filter (see key decisions).
- Not yet: contact preservation, reprojection error. Success thresholds remain `[OPEN]`.
- Render stages for precomputed tracks (implemented): `name-rendered=poses.jsonl` (through the baseline's One Euro filter) and `name-glide=poses.jsonl` (60 Hz slerp upsampling, no smoothing; `upsample.py`).

Next step: record more seated-v2 sessions with the owner (alongside M1) and measure the baseline on them. Usage and metric definitions: `eval/README.md`.

### mhr-vrm-spike (implemented, M3 done)
Location: `spikes/m3-mhr-vrm/` (uv project, CPU PyTorch, `pymomentum`, `mhr`). Converts MHR skeleton states to VRM 1.0 normalized local rotations, fits CMU mocap to MHR, and checks the round trip. Result: go for MHR as canonical skeleton (`PLAN.md` §3.10; numbers in `plans/m3-mhr-vrm-spike.md`). Its reusable code moved to `core/` (`mtcore`); the spike imports it. Data lives in gitignored `data/` (`data/cmu`, `data/mhr`). Known issue: `MHR.from_files` segfaults in pymomentum (0.1.114 CPU) when loading blendshapes; the spike loads the rig without them. Details: `modules/mhr-vrm-spike.md`.

### m1-body-lifter (WIP, M1)
Body bones only (spine, neck, head, clavicles, arms); hands, face, merger, root translation come later. Built from `core/` (`mtcore`, implemented) and `train/` (`mttrain`, implemented); CMU-derived fits, training data and checkpoints stay in gitignored `data/m1/`. Latest model: `m1-v4` (`data/m1/runs/m1-v4`; numbers in the plan, step 7b; current fits are `data/m1/fits-v3`). Resting arms now read correctly (forearm vs image 2-15 deg, was 33-42); hand position 5.2 cm raw vs baseline 3.9; MHR limit violations in 1.3% of frames by more than 1 deg (was 44%); arm rest noise still not better than the baseline (3.5 vs 3.2 deg, glide vs `baseline-rendered`). Earlier runs: much calmer shoulders/hands than the baseline with no added lag, torso/head slightly noisier. Still open: seated-v2 recordings (owner), arm rest noise, fairness of criterion 7 (the baseline solves from the same landmarks the metric uses), go/no-go write-up.

### mtcore (implemented)
Location: `core/`. Shared by the spike and `train/` as an editable path dependency; consumers pick their own torch build. Details: `modules/mtcore.md`.

### train (implemented)
Location: `train/`. Pipeline prep_fits → seated → camera → mpstyle → noise → synth → features → dataset → model → losses → train → infer, plus calibrate (real vs synthetic stats). Training runs on the RTX 4060 (`cuda:1`) with training data generated in batches on the same GPU (`gpugen`, torch port of the synthesis pipeline; the numpy generator stays as the tested reference and builds the validation set). `limitcheck` measures MHR limit violations of a checkpoint (criterion 8). Heavy CPU jobs (fits, training) run under `tools/guarded-run.sh`. Details: `modules/train.md`.

### retargeting (planned, M4)
Learned retargeting layer (`PLAN.md` §3.10a): maps the performer's pose (tracked on the performer's realistic MHR identity) onto the avatar's MHR identity, then to VRM 1.0. Avatar bone lengths enter only here. Trained to keep spatial accuracy (relative positions/directions between body parts, contacts included); may read earlier latents for temporal context. The M3 spike confirmed it is needed: proportion differences, not the conversion, cause contact misses on stylized avatars. No module file yet.

## Key decisions
- Build the evaluation harness and measure the baseline before training any model (`PLAN.md` §6).
- Baseline source is the vspark branch including uncommitted WIP, judged more accurate than its last commit (`[USER]`, 2026-11-01).
- Baseline extraction method: frozen copy (`[USER]`, 2026-11-01, `PLAN.md` §1.4).
- Eval harness in Python, uv project in `eval/` (`[USER]`, 2026-11-01).
- Test sessions record both raw webcam video and browser-side MediaPipe landmarks in one session (`[USER]`, 2026-11-01).
- First harness version: label-free metrics on real recordings; labeled metrics built and tested on synthetic signals for M1 (`[USER]`, 2026-11-01).
- MHR is the canonical skeleton; VRM poses derive from MHR rotations (`[USER]`, 2026-11-01). Validated by the M3 spike (2026-11-01).
- Performer and avatar are separate MHR identities (`[USER]`, 2026-11-01).
- M1 decisions (owner, 2026-11-01; details in `plans/m1-body-lifter.md`):
  - Training input synthesised directly from fitted MHR poses + calibrated noise and simulated camera framing; render-then-detect stays in M6.
  - Model: small stateless causal model over past samples at fixed time offsets (≈ 0.1–1.6 s) with explicit differences, 6D rotation output; preferred over a GRU (dropped frames, stateless calls, ONNX export).
  - Data: low-root-travel CMU clips weighted toward upper-body activity; seated motion from CMU upper bodies on a seated lower body; performer proportions varied within MHR limits.
  - Code layout: new `train/` uv project (CUDA PyTorch); reused spike code moved into the shared package `core/` (`mtcore`).
  - Per-bone confidence output from the start (uncertainty-weighted loss); the model predicts all four MHR spine joints, folded onto VRM's spine bones by the conversion.
- Render stage for learned-model output reuses vspark's frontend One Euro filter code via the baseline CLI, not a Python reimplementation (owner, 2026-11-07).
- M1 training data is generated in batches on the GPU (torch port), not by CPU DataLoader workers (owner, 2026-11-08): CPU generation (~4.5 ms/sample, mostly Python overhead) needed ~10 workers, pymomentum's solver threads overloaded the machine and it crashed; the model step alone is ~11.7 ms and 128 MiB on the 4060.
- M1 go/no-go treats the model as a replacement for both vspark's solver and its One Euro filter: model output goes through glide and is compared against `baseline-rendered`; model through the filter (`-rendered`) is a secondary number (owner, 2026-11-08, `PLAN.md` §6).

## Licensing
`docs/LICENSES-REGISTER.md` tracks source, license and usage of every asset, dataset, model and recording. The vendored vspark code has no declared license; the owner must relicense it before any public release.
