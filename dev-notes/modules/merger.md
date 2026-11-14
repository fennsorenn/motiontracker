# merger (`mttrain.merger`, M4 learned merger: WIP)

Location: `train/src/mttrain/merger/` (part of the `train` uv project, `modules/train.md`), plus `core/src/mtcore/facepose.py` (`modules/mtcore.md`) and `tools/merger/`. Learned step that joins the part models into one pose: the body model `m1-v10` (`modules/train.md`), the pair hand model `p3` (`modules/hands.md`) and the head/face (Holistic face mesh, MediaPipe blendshapes). Replaces the hand-written combination `mttrain.hands.wrist` (`modules/hands.md`, "Combination onto the body model"). Plan, decisions and all numbers: `plans/m4-merger.md` (Decisions, Progress); owner decisions `PLAN.md` §3.4.

Status (2026-11-14): first full models `mg1` and `mg2` trained and evaluated on all rendered sets; not in the live runtime (`service/`, follow-up plan once a checkpoint is accepted). Branch `feature/m4-merger`.

Data (gitignored): `data/merger/` (`pool/`, `calib/`, `runs/<name>/`, `eval/<run>/`); comparison videos `data/videos/<session>-<run>/`.

## Design (owner decisions, 2026-11-14)
- Scope body + hands + head/face, trained together. Part models stay frozen.
- Inputs: the part models' outputs only (rotations, log scales, hand image position/size, lost flag, detector score, face head pose, blendshapes). Internal features later, with dropout.
- Output: residual corrections on a base pose (the hand-written combination), so an untrained or switched-off merger gives a sensible pose and parts stay swappable.
- Short history of part outputs at fixed offsets (like `m1`).
- Smoothing through training: extra jitter on the inputs, jitter-free targets (low-passed idle life, no finger tremor), acceleration loss against the truth's so it filters noise without lagging.
- Blendshapes: real recorded sequences, unpaired with the motion, 30% dropout; passed through unchanged in this version.

## Pieces
| File | Role |
|---|---|
| `mtcore/facepose.py` | Head pose from Holistic's face mesh: weighted similarity fit of MediaPipe's canonical face model (33 Procrustes points), weak perspective, in image units. Metadata read from the cached Face Landmarker bundle with a minimal protobuf wire decoder; numpy-only, so `service/` can use it. `CanonicalFace`, `fit`. |
| `facecal.py` | Calibrates that head pose against the truth on the rendered sets → `data/merger/calib/face.json`: offset face frame vs MHR head joint, per-axis slope/bias, per-sequence offset sd, within-sequence sd, jitter, centre/scale models, seen rate. |
| `gapcal.py` | Body model sim-to-real gap per joint: `m1-v10` global errors on real renders vs synthetic merger batches → `data/merger/calib/m1_gap.json`. |
| `batch.py` | `MergerBatches`: training batches on the GPU (see below). Also `combine` (base pose), `degrade`, `geodesic`, `globals_from_locals` / `locals_from_globals`. |
| `model.py` | `Merger`: frame MLP over `OFFSETS` (0, 1, 2, 4, 8 frames at 10 Hz) + differences between neighbouring offsets → per body joint a 6D residual on the base pose's **global** rotation (`G = G_base R(res)`, locals follow), log scales, per-hand finger residuals and log scales. Last layer zero-initialized (untrained = base pose). `frame_inputs`, `apply`. |
| `train.py` | Training, resumable (see below). |
| `run.py` | Offline runner: landmarks file (recorded or rendered) → `{merger,base,m1}.poses.jsonl` VRM tracks (fingers in merger and base). Same chain as training; at session start the first frame is repeated. |
| `evalsummary.py` | Summary table of a rendered-set evaluation (median global error of head, arms, hands; head onset lag; body model / base / merger). |
| `tools/merger/eval.sh <run>` | Rendered-set evaluation through the harness with GT (`modules/eval-harness.md`), resumable per sequence; `SETS` overrides the set list. Output `data/merger/eval/<run>/` + `summary.md`. |
| `tools/merger/video.sh <run>` | Comparison videos webcam \| combination \| merger with face blendshapes (seated-v1, hands-v1), via `tools/avatar-video`. Needs `data/videos/<session>-full/face.landmarks.jsonl` (runtime dump with face) and a vspark checkout next to the repo. |

## Batches (`MergerBatches`)
- Two sources of full MHR parameter crops at 30 fps: (a) `gpugen` crops (CMU seated, choreography, head moves) with `Seater` `smooth_life` and procedural fingers from a finger bank (`hands.motion`, tremor off); (b) the merger pool `data/merger/pool` (`hands.pool2 --full --no-tremor`: hands in use, rests, body contact, two-hand events) with head life and head moves added. Pool 2026-11-14: 1228 train / 93 val sequences of 20 s.
- One shared camera; detector frames at 10 Hz; K = 11 target frames.
- Body: `GpuGenerator.observe` → frozen `m1-v10`. Hands: `observe2.PairObserver` on the same camera → frozen `p3`. Head: simulated face-mesh pose from the true head with the `facecal` error model (per-axis slope, per-sequence offset, slow drift + white jitter, lost runs). Blendshapes: real, from `data/videos/*-full/face.landmarks.jsonl`.
- Extra input jitter (`--jitter-deg`, default 0.75 deg): small because synthetic part outputs already jitter about as much as real ones at rest.
- `degrade`: the body model's outputs pushed to their real error level (per-joint global noise from `m1_gap.json`, 70% per-sequence offset, rest drifting). Without it the merger trusted the body model too much (pilot 1); `--no-gap` turns it off.
- Clipping correction: 30% of samples get observed hands near the torso/thighs pushed 1-4 cm into the body; the target stays outside.
- Base pose `combine`: in the body model's rest-removed camera frame, wrist global = `F^T F_rest` from the hand model's palm frame, head from the face pose (`R_face offset^T rest_head^T`), rest from the body model.

## Training (`train.py`)
`uv run python -m mttrain.merger.train --name <run> [--steps 20000] [--resume]` (under `tools/guarded-run.sh`, `modules/train.md`). Runs in `data/merger/runs/<name>`; `last.pt` holds optimizer, scheduler, generator and torch RNG state, `best.pt` by the sum of wrist/head global, forearm, head and in-view finger errors on fixed validation batches (merger vs base pose vs body model).
- Losses: `m1`'s body losses (`mttrain.losses`: rotation with Laplace scales, FK positions, velocity, limits, contacts), global wrist/head angle, finger Laplace NLL (in and out of view), acceleration of FK positions and fingers against the truth's. Evaluated at the newest three frames.
- Flags: `--jitter-deg`, `--no-gap`, `--w-con` (contact), `--w-glob`, `--w-fing`, `--w-acc`, `--pool`.
- 20000 steps take ~50 min on the RTX 4060.

## Results so far (details in the plan, Progress)
- `mg1` (pilot-3 setup) and `mg2` (`--w-con 0.5 --w-glob 3`) are about level.
- Rendered sets vs body model and base pose: lower arms better nearly everywhere, head better on most sets (slightly worse on rest1, v1), hands better on pairs3, level on hands3/rest1, worse on contact1, head1 and v1.
- Open: on contact1 the merger overrides even confident hand-model wrists (not the contact loss, see `mg2`). Suspects: training data near the face (only from the body model's choreography source, procedural fingers) and the hand model's confidence calibration on synthetic data.

## Shared-code changes made for the merger
- `gpugen.GpuGenerator.sample` split into `crop` / `observe` (m1 batches bit-identical), optional landmark `push` hook; `gpugen.motion.Seater` `smooth_life` (`modules/train.md`).
- `hands`: `observe2.PairObserver` optional camera; `HandMotion` tremor flag; `pool2 --full` (full MHR sequences) / `--no-tremor`; `evaluate2.LearnedPair` exposes raw outputs (`raw`, `lost`, `score`) (`modules/hands.md`).

## Extending
- Anything the runtime will need (face pose, feature construction) goes into `mtcore` and stays numpy-only.
- A new part model or input: add it to `batch.py` (observed from the same camera and crop), to `frame_inputs`, and to `run.py` so offline inference follows the training chain; keep the base pose sensible without the merger.
- Re-run `gapcal` / `facecal` when the body model or detector setup changes; the batches read their JSON.

## Open
- The contact1 wrist override (above); hands behind the base pose on head1 and v1.
- Runtime integration (`service`), internal features as inputs, joint fine-tuning, gaze, a learned face model: out of scope for this plan.
