# Architecture

Learned live motion tracking for VRM 1.0 avatars: webcam → MediaPipe detectors → small causal learned heads + merger → bone rotations, root translation, blendshapes, gaze → VMC/OSC. Full context, decisions and open questions: `PLAN.md` (decision tags there are authoritative; `[CLAUDE]` items are unconfirmed).

No learned-model or service code exists yet. Current milestone: the M3 MHR ↔ VRM 1.0 mapping spike, run before M1 (owner decision 2026-11-01, `PLAN.md` §10 and §9 #1). M0 tooling is done and the first real session (seated-v1) has been measured. Remaining M0 items are deferred: more seated-v2 recordings run alongside M3, and the face-cost check moves to M4.

## Modules

| Module | Status | Summary |
|---|---|---|
| `baseline` | implemented (frozen) | Frozen snapshot of the owner's current algorithmic MediaPipe → VRM bone solver (vspark), for comparison. Details: `modules/baseline.md`. |
| `eval-harness` | implemented | Python (`eval/`, package `mteval`): runs systems over recorded sessions and computes metrics (`PLAN.md` §6). Details: `modules/eval-harness.md`. |
| `recorder` | implemented | Static browser page (`tools/recorder/`) that records test sessions: raw video + MediaPipe landmarks + protocol segments. Details: `modules/recorder.md`. |
| `mhr-vrm-spike` | WIP (M3) | Spike: MHR ↔ VRM 1.0 joint mapping and round-trip check. |
| `retargeting` | planned (M4) | Learned layer mapping the performer's pose onto the avatar's MHR identity, then to VRM 1.0 (`PLAN.md` §3.10a). |

### baseline (implemented, frozen)
Location: `baseline/` (own pnpm/TypeScript package). Byte-identical copies of vspark's solver nodes under `baseline/vendor/`, driven by a minimal standalone pipeline (`src/pipeline.ts`) instead of vspark's graph engine. CLI: MediaPipe landmark frames JSONL in → `OutputFrame` JSONL out (`{t, bones:{vrmBone:[x,y,z,w]}}`). Verified bit-exact against vspark's real engine. Usage, provenance and fidelity quirks: `baseline/README.md`.

- Frozen: the reference the learned pipeline must beat. Never edit `vendor/` to improve it.
- Notable reproduced quirk: vspark holds the last hand/face landmarks when a stream drops out (on by default, `holdMissing`).

### recorder (implemented)
Location: `tools/recorder/`. Reproduces vspark's CameraCapture path (mirrored, 320x240, 10 FPS, MediaPipe tasks-vision 0.10.17 HolisticLandmarker, VIDEO mode). Writes one session directory (`session.json` + `landmarks.jsonl` + `video.webm`) into gitignored `sessions/`. Can also replay a video file instead of the webcam. Usage and fidelity notes: `tools/recorder/README.md`.

### eval-harness (implemented)
Location: `eval/` (uv project, package `mteval`, CLI `mteval evaluate`). Runs systems over a session (baseline via its CLI at solver and rendered stages; any other system from a precomputed `OutputFrame` JSONL) and writes a JSON + markdown report. All metrics are time-based.
- Label-free metrics on real recordings: rest jitter/noise, twitch rate, speed percentiles, hand-dropout peaks, lag vs detector, presence.
- Labeled metrics (rotation error, onset lag, peak-velocity ratio): tested on synthetic signals only, not in the report until labeled data exists (M1).
- Compare systems at the same stage only (solver vs solver, rendered vs rendered).
- Not yet: contact preservation, reprojection error. Success thresholds remain `[OPEN]`.

Next step: record more seated-v2 sessions with the owner (alongside M3) and measure the baseline on them. Usage and metric definitions: `eval/README.md`.

### mhr-vrm-spike (WIP, M3)
Started 2026-11-01. Scope: load the MHR rig and list its joints; build an MHR → VRM 1.0 joint mapping table; round-trip VRM → MHR → VRM on CMU clips; fit the VRoid sample avatars as MHR identities. Check spine, twist joints, fingers/thumb and rest pose. Code location and outcome not yet known. No module file yet.

### retargeting (planned, M4)
Learned retargeting layer (`PLAN.md` §3.10a): maps the performer's pose (tracked on the performer's realistic MHR identity) onto the avatar's MHR identity, then to VRM 1.0. Avatar bone lengths enter only here. Trained to keep spatial accuracy (relative positions/directions between body parts, contacts included); may read earlier latents for temporal context. Comes after the M3 MHR↔VRM mapping spike. No module file yet.

## Key decisions
- Build the evaluation harness and measure the baseline before training any model (`PLAN.md` §6).
- Baseline source is the vspark branch including uncommitted WIP, judged more accurate than its last commit (`[USER]`, 2026-11-01).
- Baseline extraction method: frozen copy (`[USER]`, 2026-11-01, `PLAN.md` §1.4).
- Eval harness in Python, uv project in `eval/` (`[USER]`, 2026-11-01).
- Test sessions record both raw webcam video and browser-side MediaPipe landmarks in one session (`[USER]`, 2026-11-01).
- First harness version: label-free metrics on real recordings; labeled metrics built and tested on synthetic signals for M1 (`[USER]`, 2026-11-01).
- MHR is the canonical skeleton; VRM poses derive from MHR rotations (`[USER]`, 2026-11-01).
- Performer and avatar are separate MHR identities (`[USER]`, 2026-11-01).

## Licensing
`docs/LICENSES-REGISTER.md` tracks source, license and usage of every asset, dataset, model and recording. The vendored vspark code has no declared license; the owner must relicense it before any public release.
