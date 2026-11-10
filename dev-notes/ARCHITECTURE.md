# Architecture

Learned live motion tracking for VRM 1.0 avatars: webcam → MediaPipe detectors → small causal learned heads + merger → bone rotations, root translation, blendshapes, gaze → VMC/OSC. Full context, decisions and open questions: `PLAN.md` (decision tags there are authoritative; `[CLAUDE]` items are unconfirmed).

No service code exists yet. Current milestone: **M1, the body lifter in MHR space** (WIP, plan: `plans/m1-body-lifter.md`): pipeline, GPU training-data generator, render stage, criterion 7/8 checks, coherent arm noise, rendered synthetic sessions, MediaPipe depth flattening in the synthesis and designed contact choreography done, contact-aware training (`m1-v8`) done, current best model `m1-v7` (`m1-v8` level elsewhere, better at contacts); the go/no-go still needs seated-v2 sessions. Done: M0 tooling plus the first measured session (seated-v1), and the M3 MHR ↔ VRM 1.0 spike (result: go for MHR as canonical skeleton). The face-cost check moves to M4.

## Modules

| Module | Status | Summary |
|---|---|---|
| `baseline` | implemented (frozen) | Frozen snapshot of the owner's current algorithmic MediaPipe → VRM bone solver (vspark), for comparison. Details: `modules/baseline.md`. |
| `eval-harness` | implemented | Python (`eval/`, package `mteval`): runs systems over recorded sessions and computes metrics (`PLAN.md` §6). Details: `modules/eval-harness.md`. |
| `recorder` | implemented | Static browser page (`tools/recorder/`) that records test sessions: raw video + MediaPipe landmarks + protocol segments. Details: `modules/recorder.md`. |
| `mhr-vrm-spike` | implemented (M3, done) | Spike (`spikes/m3-mhr-vrm/`): MHR → VRM 1.0 conversion, CMU → MHR fitting, round-trip eval. Reusable code now in `core/`. Details: `modules/mhr-vrm-spike.md`. |
| `m1-body-lifter` | WIP (M1) | First learned model: MediaPipe pose landmarks → small causal model → MHR upper-body rotations → VRM 1.0. Go/no-go for the learned approach. Plan: `plans/m1-body-lifter.md`. |
| `mtcore` | implemented | Shared Python package (`core/`): MHR loading/FK, CMU reader + CMU → MHR fit, MHR → VRM conversion, MediaPipe landmarks from MHR. Details: `modules/mtcore.md`. |
| `train` | implemented | uv project (`train/`, package `mttrain`, CUDA PyTorch): CMU fits, seated synthesis, synthetic landmarks + noise, batched GPU data generator, contact choreography, model, training, inference on sessions, limit check. Details: `modules/train.md`. |
| `render` | implemented | Rendered synthetic sessions with exact ground truth: `mttrain.render` + `tools/render/render_blender.py` (Blender EEVEE) → video → recorder replay → real MediaPipe landmarks + `gt.jsonl`. Details: `modules/train.md` (Rendered sessions). |
| `mp-depth-probe` | implemented (spike, done 2026-11-09) | Spike (`spikes/mp-depth-probe/`): where MediaPipe's pose landmark model encodes hand-vs-face depth. Result: both landmark heads are linear in one 768-d feature (`re_lu_55`) with no linear trace of that depth (CV R2 ~0); only a weak mid-network trace (R2 <= 0.29, 16x16-8x8 stages); MediaPipe's output moves the wrong way as the hand leaves the face. A closed-form head edit is ruled out; a fine-tune would have to reach the mid network. More promising cue: hand size relative to face size (Holistic hand/face landmarks) fed to the lifter. Renders only; re-check on real frames once two-camera depth exists. Details: `spikes/mp-depth-probe/README.md`. |
| `avatar-video` | implemented | Dev tool (`tools/avatar-video/`): side-by-side comparison video of a recorded session on a VRM avatar (mirrored webcam \| `baseline-rendered` \| a learned model's glide track). Output in gitignored `data/`. Usage: `tools/avatar-video/README.md`. |
| `retarget-upper-bound` | implemented (spike, done 2026-11-10) | Spike (`spikes/retarget-upper-bound/`, run from `train/`'s uv env): best-case contact gain of a retargeting layer, plain rotation copy (`mtcore` `MhrToVrm`) vs IK onto the avatar's own face / other palm. Result on `contact1` true poses and 11 VRoid avatars: copy puts the hand ~7 cm inside the head at hand-to-face contacts (9 cm on male-base rigs) and collapses 6 cm near misses to touching; IK fixes it but needs ~22 deg forearm / ~39 deg hand change; palms survive copy within ~3 cm. So retargeting matters at hand-to-face, must allow large pose changes near the face, and needs contact intent from the lifter (blocked today by MediaPipe's near-face depth bias). Side finding: `mttrain.choreo` contact poses penetrate the skin by 2-4 cm (targets joint centres). Details: `spikes/retarget-upper-bound/README.md`. |
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
- Label-free metrics on real recordings: rest jitter/noise, twitch rate, speed percentiles, hand-dropout peaks, lag vs detector, presence, criterion 7 as restated 2026-11-09 (2D wrist error in the camera plane, forearm direction, contact segments).
- Ground-truth metrics (`--gt gt.jsonl`, for rendered sessions): global rotation error, onset lag, peak-velocity ratio, wrist error; report section "Against ground truth".
- Compare systems at the same stage only (solver vs solver, rendered vs rendered). A learned model counts as rendered when shown through glide or the filter (see key decisions).
- Not yet: reprojection error. Success thresholds remain `[OPEN]`.
- Render stages for precomputed tracks (implemented): `name-rendered=poses.jsonl` (through the baseline's One Euro filter) and `name-glide=poses.jsonl` (60 Hz slerp upsampling, no smoothing; `upsample.py`).

Next step: record more seated-v2 sessions with the owner (alongside M1) and measure the baseline on them. Usage and metric definitions: `eval/README.md`.

### mhr-vrm-spike (implemented, M3 done)
Location: `spikes/m3-mhr-vrm/` (uv project, CPU PyTorch, `pymomentum`, `mhr`). Converts MHR skeleton states to VRM 1.0 normalized local rotations, fits CMU mocap to MHR, and checks the round trip. Result: go for MHR as canonical skeleton (`PLAN.md` §3.10; numbers in `plans/m3-mhr-vrm-spike.md`). Its reusable code moved to `core/` (`mtcore`); the spike imports it. Data lives in gitignored `data/` (`data/cmu`, `data/mhr`). Known issue: `MHR.from_files` segfaults in pymomentum (0.1.114 CPU) when loading blendshapes; the spike loads the rig without them. Details: `modules/mhr-vrm-spike.md`.

### m1-body-lifter (WIP, M1)
Body bones only (spine, neck, head, clavicles, arms); hands, face, merger, root translation come later. Built from `core/` (`mtcore`, implemented) and `train/` (`mttrain`, implemented); CMU-derived fits, training data and checkpoints stay in gitignored `data/m1/`. Current best model: `m1-v7` (`data/m1/runs/m1-v7`, depth flattening with wide camera sampling; `m1-v5`/`m1-v6` kept; contact-aware `m1-v8` level elsewhere, better at contacts; numbers in the plan, steps 7b-7e; current fits are `data/m1/fits-v3`). Resting arms now read correctly (forearm vs image 2-15 deg, was 33-42); hand position 5.2 cm raw vs baseline 3.9; MHR limit violations in 1.3% of frames by more than 1 deg (was 44%); arm rest noise still not better than the baseline (3.5 vs 3.2 deg, glide vs `baseline-rendered`). Earlier runs: much calmer shoulders/hands than the baseline with no added lag, torso/head slightly noisier. Rendered sets `v1` (motion, 8 x 20 s), `rest1` (6 x 15 s) and `fov1` (3 clips x 4 FOVs) exist. Key finding (step 7c): MediaPipe's world landmarks flatten depth (toward-camera slope ~0.42 upper arm, ~0.32 forearm vs truth; image plane ~1), and the models inherited it. Still open: seated-v2 recordings (owner; a two-camera multi-view setup is in progress, also needed to measure real flattening strength), arm rest noise, go/no-go write-up, and an owner decision on smoothing the seated idle motion (rough AR(1), so rendered rest ground truth is never still).
- MediaPipe depth flattening in the synthesis (implemented, commits 1913fdb, d437e72; owner decision 2026-11-09: train the model to undo the flattening): synthetic world and image landmarks get MediaPipe's per-segment depth compression measured on rendered sessions (depth ≈ gain × true + bias, segment lengths kept; image z ≈ 3× world z), stronger for closer cameras, and relaxed per sequence toward none because real video flattens less than the renders. Cameras sampled widely (FOV 50-130 deg, 0.45-1.6 m). Details and limits: `modules/train.md`.
- Designed contact choreography (implemented, commits 546e137, 8fffdcf): `mttrain.choreo` builds seated sequences with exact hand-hand / hand-face contact intervals via MHR IK; used for the rendered contact set `contact1` and as a training pool. On `contact1` every system leaves the hand 11-14 cm too far from the face; palms together `m1-v7` is best (3.9 cm). Details: `modules/train.md`.
- Contact-aware training (implemented, commit 0ed4a2b): choreography pool mixed into training (`--p-choreo`, default 0.2) plus a contact loss on contact-point differences. First model `m1-v8`: hand-to-face on `contact1` 11.1 → 7.9 cm, palms on the true body 10.0 → 6.8 cm (on MediaPipe's skeleton 3.9 → 4.9), about level with `m1-v7` elsewhere (plan step 7e). Open: MediaPipe places a hand touching the face 8-26 cm in front of the nose, like a real near-miss; the synthesis lacks this bias, so all models put the hand ~30 cm toward the camera. Unverified on real video (needs seated-v2 contact segments). Details: `modules/train.md`.

### mtcore (implemented)
Location: `core/`. Shared by the spike and `train/` as an editable path dependency; consumers pick their own torch build. Details: `modules/mtcore.md`.

### train (implemented)
Location: `train/`. Pipeline prep_fits → seated → camera → mpstyle → noise → synth → features → dataset → model → losses → train → infer, plus calibrate (real vs synthetic stats). Training runs on the RTX 4060 (`cuda:1`) with training data generated in batches on the same GPU (`gpugen`, torch port of the synthesis pipeline; the numpy generator stays as the tested reference and builds the validation set). `limitcheck` measures MHR limit violations of a checkpoint (criterion 8). Heavy CPU jobs (fits, training) run under `tools/guarded-run.sh`. Details: `modules/train.md`.

### avatar-video (implemented)
Location: `tools/avatar-video/`. Side-by-side comparison video of a recorded session: mirrored webcam video next to one avatar panel per system (front view plus side-view inset). Usage: `tools/avatar-video/README.md`.
- `prepare.py` (run with `uv run --project eval`) builds `tracks.json` from harness system specs (e.g. `baseline-rendered`, `m1-v8-glide=...`). Limb bones only where all systems output them (no fingers on either side); the torso/neck/head chain is kept per system (the baseline has no hips/upperChest/head).
- `render.mjs` drives headless Chrome (three.js 0.168.0 + @pixiv/three-vrm 3.5.2 from jsDelivr, `setNormalizedPose` like vspark's viewport; VRM 0.x x/z negated after `rotateVRM0`); ffmpeg hstacks the panels with the webcam video. About 7 frames/s on 8 cores; run under `tools/guarded-run.sh`.
- Not reproduced from vspark: arm calibration, tracking-loss fades, root motion, IK mode. No retargeting.
- Output in gitignored `data/videos/` (contains the owner's webcam video), never committed. First output: `data/videos/seated-v1-m1-v8/compare.mp4` (seated-v1, AvatarSample_F, `baseline-rendered` vs `m1-v8` glide).

### retargeting (planned, M4)
Learned retargeting layer (`PLAN.md` §3.10a): maps the performer's pose (tracked on the performer's realistic MHR identity) onto the avatar's MHR identity, then to VRM 1.0. Avatar bone lengths enter only here. Trained to keep spatial accuracy (relative positions/directions between body parts, contacts included); may read earlier latents for temporal context. The M3 spike confirmed it is needed: proportion differences, not the conversion, cause contact misses on stylized avatars. No module file yet.
- Upper-bound spike `retarget-upper-bound` (done 2026-11-10): large gain at hand-to-face (~7 cm penetration under copy), small at palms; the layer cannot be a small residual near the face and needs contact intent from the lifter. See the module table row and the spike README.

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
- Criterion 7 restated (owner, 2026-11-09, `PLAN.md` §6): 2D wrist error in the camera plane against the image landmarks (primary), forearm direction as a guard, 3D on-skeleton error secondary, plus contact segments (7b).
- MediaPipe's depth flattening is modelled in the synthesis so the model learns to undo it, rather than waiting for multi-view data or render-then-detect training (owner, 2026-11-09). Camera FOV sampled up to 130 deg (owner).
- Rendered evaluation sessions use Blender (owner, 2026-11-09; resolves `PLAN.md`'s three.js-vs-Blender item for rendered evaluation). Renders are CMU-derived: `data/m1/renders`, never committed.

## Licensing
`docs/LICENSES-REGISTER.md` tracks source, license and usage of every asset, dataset, model and recording. The vendored vspark code has no declared license; the owner must relicense it before any public release.
