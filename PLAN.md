# Learned Live Motion Tracking for VRM Avatars — Project Plan

Status: planning complete, no code written yet.
Origin: a long brainstorming conversation between the project owner and Claude (claude.ai, Nov 2026). This document captures everything from that discussion so work can continue in this repo.

---

## 0. How to read this document

Every decision in this plan is tagged with who made it. **This matters:** only items tagged `[USER]` are confirmed. Everything else is a proposal that can and should be challenged.

| Tag | Meaning |
|---|---|
| `[USER]` | Decided or explicitly stated by the project owner. Treat as a requirement unless the owner changes it. |
| `[USER-LEAN]` | Owner's stated preference, not final. Confirm before building on it heavily. |
| `[CLAUDE]` | Recommended by Claude during planning. **Not confirmed by the owner.** A starting point, open to revision, especially if evidence contradicts it. |
| `[OPEN]` | Unresolved. Needs a decision, usually after a spike or measurement. |
| `[DISCARDED]` | Considered and rejected (with reason). Don't re-propose without new information. |
| `[CORRECTED]` | Something Claude said earlier that was later found to be wrong and retracted. |

Where an idea originated with the owner but the specific design details came from Claude, both are noted.

---

## 1. Goal and requirements

### 1.1 What we're building
A live motion-tracking system that drives VRM avatars (primarily VRoid models) from a webcam: head, upper body (mostly), arms and hands, plus face blendshapes for completeness. `[USER]`

### 1.2 The problem with the current approach
- The current pipeline uses MediaPipe and works "mostly", but is jittery. The jitter is amplified by the landmark → skeleton (bone rotation) translation step. `[USER]`
- The owner has already exhausted what an algorithmic MediaPipe pipeline offers (filtering, solver tuning, etc.) and is not satisfied with the results. `[USER]`
- So the core idea is a **learned** stage that turns detector output into stable skeleton rotations directly. `[USER]` (the owner asked whether a model could be tuned/retrained to output more stable results and directly output skeleton pose; the specific lifter design is `[CLAUDE]`, see §3)

### 1.3 Hard requirements
- **Latency is critical**; this is for real-time use. `[USER]`
- **Lightweight is a strong criterion**: it must run on already busy hardware. `[USER]`
- **Hardware target:** "runs on anything that runs MediaPipe in the first place". Should be feasible on **integrated graphics**, possibly with only one or two tracking types enabled. `[USER]`
- **Dev hardware:** RTX 4060 and RTX 3090 available for training/dev. `[USER]`
- **Licensing:** owner wants to release under a **permissive license**. All data, models and assets must allow that. Training data must come from **permissively licensed** sources. `[USER]`
- **No owner-provided marker/mocap data.** At most: limited, low-quality, unlabeled video. `[USER]`
- **Target avatar format: VRM 1.0.** Retargeting to VRM 0.x when needed is considered straightforward. `[USER]`
- **Model vendors:** prefers models from one vendor, but open to mixing per tracking type. `[USER]`

### 1.4 Context about the existing app
- The app runs in **Node**. Its stack is based on **three.js** (three-vrm). `[USER]`
- **ARKit** is already a tracking option in the app, but only covers face/head and isn't available to all users. `[USER]`
- The current tracking implementation exists on the owner's machine and will be **extracted as the baseline** for comparison. `[USER]`
  - Source: `../vspark`, branch `bugfix/mediapipe-arm-solve` **including its uncommitted WIP** (elbow-hinge arm solve + confidence-weighted `pose_smooth` node), judged more accurate than the last commit. `[USER]` (2026-11-01)
  - Extraction method: **frozen copy** of the solver into `baseline/` with a standalone driver (not running vspark in place, not recorded outputs only). `[USER]` (2026-11-01). Provenance and fidelity notes in `baseline/README.md`.

---

## 2. Research findings (as of Nov 2026)

These are facts found during planning, with sources. Verify licenses again before release; Claude is not a lawyer.

### 2.1 Comparable projects
- **MocapNET** (FORTH, Qammaz & Argyros): closest prior art. Estimates 3D pose directly in BVH format from 2D joint estimates, runs real-time on CPU, uses an ensemble with the body split into upper/lower hierarchies for occlusion robustness. v4 adds face, gaze and hands. Custom license (not standard OSS). → Use as **reference only**, not a dependency. https://github.com/FORTH-ModelBasedTracker/MocapNET
- **XR Animator**: best-known webcam → VRM/MMD app. MediaPipe + hand-written solver (the kind of pipeline the owner has outgrown). Licensed **CC BY-NC-SA 4.0** → not usable as code.
- No one was found shipping a learned, temporal keypoints → VRM rotations model. This project isn't duplicating existing work.

### 2.2 Base detection models
- **MediaPipe** (Google): models are licensed **Apache 2.0** per Google's model cards. Pose landmarker lite/full/heavy ≈ 2.9 / 6.3 / 27 MB. Pose model is a MobileNetV2-like CNN on a 256×256 ROI crop, fed by a detector and then tracked from the previous frame's auxiliary landmarks.
  - Pose model has been converted TFLite → ONNX → PyTorch with numerical parity by third parties (e.g. https://huggingface.co/mickeyvanolst/blazepose-coreml documents the recipe). **So feature taps and fine-tuning are possible.**
  - **Face Landmarker** is one pipeline: face detector → face mesh (478 landmarks) → **blendshape model**. The blendshape model is a tiny MLP-Mixer that takes 146 landmarks (never sees pixels) and outputs 52 ARKit-compatible blendshapes; runs well under 1 ms on CPU. Ported to PyTorch (py-feat) and ONNX (with a claim of Apache 2.0 weights; verify against the official model card).
  - Running mode defaults to **IMAGE**, which runs the detector every frame (no tracking). VIDEO / LIVE_STREAM enable tracking and are much cheaper.
- **RTMPose** (OpenMMLab): code Apache 2.0, full training code. Small variants: RTMPose-t ≈ 0.36 GFLOPs, -s ≈ 0.68 GFLOPs at 256×192. **Weights are trained on mixed datasets, some with non-commercial annotations** (e.g. CrowdPose), so weight licensing is murky. A permissive release would mean retraining on COCO-only data.
- **RTMW / RTMW3D** (whole-body, 133 keypoints): RTMW3D-L ≈ 65M params, ≈30 fps on an RTX 3090 (per a community package). Too heavy for the target. Also trained on mixed-license data.
- Other models considered: see §8 (discarded).

### 2.3 Datasets and assets
| Source | License / terms | Notes |
|---|---|---|
| **CMU Graphics Lab Mocap** | Free for research; may be included in commercially sold products; may **not resell the data**, even converted. | ~2,500 clips, mostly full-body (locomotion, sports, etc). Use the **original** or the **cgspeed BVH conversions**. At least one popular FBX conversion (4TU.ResearchData) is relicensed **CC BY-NC**, avoid it. Older/low subject numbers are lower quality; skeletons vary per subject. |
| **UPenn Multi-Modal Mocap Library** | Same terms as CMU. | Second source. |
| **100STYLE** | **License not confirmed** in research (Claude recalled CC BY 4.0, unverified). | Stylized locomotion only, single actor. Low priority. |
| **MHR (Meta Momentum Human Rig)** | **Apache 2.0** (released Nov 2025). | Parametric body model, decoupled skeleton and surface, identity (45 params) / pose (204 params) / face expression, separate body/head/hand shape spaces. Ships FBX rigs per LOD. https://github.com/facebookresearch/MHR |
| **VRoid Studio** | Models you create are yours; you set their terms. pixiv/third-party preset assets carry their own conditions. | Use for stylized synthetic characters and Perfect Sync (ARKit blendshape) avatars. |
| **MakeHuman assets** | Believed CC0 (**verify**). | Realistic human meshes for rendering. |
| AMASS, SMPL/SMPL-X, LaFAN1, Bandai Namco, OpenPose | Non-commercial / research-only. | **Excluded.** See §8. |

- MediaPipe's own hand paper found **real + synthetic training data combined beats either alone** (supports the rendering plan in §5.3).
- Meta's **SAM 3D Body** is built on MHR with separate body/hand decoders (architecturally similar to this plan). Possible offline teacher later, **but check its license first**.

### 2.4 Standards
- **VRM 1.0** humanoid derives from Unity Mecanim humanoid; normalized T-pose with identity rest rotations, clean target for learned rotations.
- VRM 1.0 **look-at** can be **bone-based** (eye bones) or **expression-based** (lookUp/Down/Left/Right) depending on the avatar.
- **OpenXR hand joint set** (26 joints, also WebXR/Quest): closest thing to a cross-platform hand standard.
- **VMC protocol**: OSC over UDP, de facto VTuber standard (VSeeFace, Warudo, VNyan…). Carries humanoid bone transforms and blendshapes. Uses **Unity left-handed coordinates**.

---

## 3. Core approach

### 3.1 Overall pipeline
```
webcam → [MediaPipe detectors: pose / hands / face]   (frozen, existing cost)
       → per-part heads (temporal, small)            (new, learned)
       → merger (thin, residual corrections)         (new, learned)
       → canonical skeleton rotations + root + blendshapes + gaze
       → retarget adapter → VRM 1.0 (→ VRM 0.x)
       → service output (VMC/OSC over UDP + WebSocket)
```
- Keep the detector frozen and add a learned **lifter** on top that outputs rotations directly. `[CLAUDE]` (owner's goal was direct skeleton output; this structure is Claude's proposal)
- Rotation representation: **6D rotations** (better for training than Euler/quaternions). `[USER]` (2026-11-01, for M1)
- Strictly **causal** (no future frames). `[CLAUDE]`, consistent with `[USER]` latency requirement.

### 3.2 Base detector: MediaPipe vs RTMPose
- Owner is **unsure** which is the better base. `[USER]` → `[OPEN]` for the owner
- Claude's recommendation: **MediaPipe**, after research: Apache-licensed weights, tiny models, iGPU-friendly, convertible to PyTorch for feature taps/fine-tuning. RTMPose as fallback if body accuracy disappoints (would need COCO-only retraining for a permissive release). `[CLAUDE]`
- **MediaPipe is the base detector.** `[USER]` (2026-11-01): no better option.
- `[CORRECTED]` Claude initially recommended **RTMW** as a single-vendor whole-body option; retracted after finding its size (~65M params) and murky weight licensing.
- `[CORRECTED]` Claude initially said MediaPipe was a "dead end" for modification (no training code). Retracted: third-party TFLite→PyTorch conversions with parity exist.
- Hybrid (RTMPose body + MediaPipe hands) was suggested at one point `[CLAUDE]`; superseded by the MediaPipe recommendation.

### 3.3 Architecture: split part heads + merger
- **Split per-part heads + a merger.** Owner proposed "split heads plus a merger head"; `[USER]` (idea), Claude agreed and detailed it.
- Parts: body, left hand, right hand, head/face. `[CLAUDE]`
- **Merger's job:** resolve consistency at joints where parts meet. Key cases: **forearm twist** (body sees elbow/wrist positions, hand sees palm orientation; only together do they determine twist) and **neck vs head** split. `[CLAUDE]`
- **Disabling parts to save compute must work** (owner's question). Approach: **modality dropout** in training: randomly drop whole parts, replace with learned "missing" token + mask flag. Same mechanism handles hands leaving frame / detector dropouts. `[CLAUDE]`
  - Note: disabling a part saves *detector* compute; the heads themselves are negligible.
- **Arms resting below the frame must look natural.** This is a normal streaming posture, not an edge case: elbows and wrists out of frame, hands undetected. `[USER]` (2026-11-01). Observed in the first real session: the baseline jitters most here because MediaPipe guesses the off-frame joints and vspark holds stale hand landmarks.
- **Hand position parity must improve.** The baseline's hands land away from where the real hands are, because per-bone rotation errors add up along the shoulder → upper arm → forearm → hand chain. `[USER]` (2026-11-01). Target: **hand position relative to the character** (contacts like hand-on-face, hands together, hand on chest preserved on the avatar's proportions) matters more than matching the hand's absolute position in space, which is approximate anyway. `[USER]` (2026-11-01). Not just contacts: relative positions and directions between parts must hold before they touch (e.g. two index fingers approaching each other). Handled by the learned retargeting layer (§3.10a). `[USER]` (2026-11-01)
- **No impossible poses**, especially no jittering into them at low confidence or on full tracking loss. `[USER]` (2026-11-01).
- **Asynchronous update rates** per part (e.g. hands at lower rate when compute-tight): each head receives time since its last update. `[CLAUDE]`

### 3.4 Head ↔ merger interface
- Heads output **rotations + per-bone confidence**, optionally a small latent. Merger predicts **residual corrections** to head rotations. `[USER]` (2026-11-14: corrections on top of the combined head outputs, `mttrain.hands.wrist`'s combination as the starting point)
- **First merger (M4, 2026-11-14): body + hands + head/face.** Joins the body model (`m1`) and the pair hand model (`p3`): forearm twist, wrist, body pose preferred where hand confidence is low, hands/arms kept out of the body (the correction data of §5.6); plus neck/head and the face (blendshapes as input). `[USER]`
- **Merger design (2026-11-14):** head seen as a head pose fitted to Holistic's face mesh (canonical face model), simulated in training with an error model calibrated on renders; blendshape input trained on real, unpaired recorded sequences with dropout (passed through unchanged); short history of part outputs; trained on a new combined pool (body sources + hand events, full MHR sequences) with clipping-correction samples. `[USER]` Plan: `dev-notes/plans/m4-merger.md`.
- **Runtime (2026-11-14):** `mg3` goes live first, as ONNX + numpy (no torch/pymomentum). `[USER]` Details §7.6.
- **Merger smoothing through training (2026-11-14):** extra jitter on the merger's inputs (calibrated on real part-model noise), jitter-free targets (smooth idle, no finger tremor, sharp motion kept). `[USER]` (idea); calibration and details `[CLAUDE]`.
- **Merger inputs, first version: head outputs only** (rotations, confidences, the hand model's image position and size); internal features follow as a second step with feature dropout. `[USER]` (2026-11-14)
  - Reasons: swappable sources (e.g. ARKit can plug in as an alternative head module), staged training, graceful degradation if merger is off, confidence-weighted fusion at shared joints.
  - Owner noted swapped-in sources could still get a trained **adapter** even with a latent interface; accepted the point. `[USER]`
- **Merger taps heads' internal layers** (pooled penultimate features) in addition to rotations. Owner's idea `[USER]`; Claude agreed.
  - Train with **feature-path dropout** so "rotations only" stays sufficient (enables adapters for ARKit etc). A fixed-size projection per head acts as a built-in adapter slot. `[CLAUDE]`
- **Training order:** heads first → merger with heads frozen → optional short joint fine-tune at low LR. **Keep head rotation losses active during joint fine-tune** so head outputs stay meaningful. `[CLAUDE]`
- Latent-only interface: see §8 (discarded as the primary interface).

### 3.5 Context fed to heads
- **Part heads get full-pose context.** Owner argued this (example: a falling body should default to open hands even when hand tracking is weak). `[USER]`
  - Design detail: feed a **compact embedding** (a few dozen numbers) of the previous merged pose and its motion, not all raw bones; mask it in training. `[CLAUDE]`
- Context is the **previous frame's merged output** (current frame's merged pose doesn't exist yet when heads run). `[CLAUDE]`
- Risks and mitigations `[CLAUDE]`:
  - **Exposure bias**: train on own rollouts, or inject realistic noise into fed-back pose.
  - **Copying** ("repeat last frame" = smooth-but-late): randomly drop feedback during training; velocity loss.

### 3.6 Temporal design
- **Temporal state lives in the per-part heads**; merger is thin (may read heads' hidden states); `[USER]` (2026-11-14): the merger gets a short history of the part outputs. Parts move at different speeds and need different smoothing. `[CLAUDE]` (owner asked whether temporal data goes into heads or merger)
- **Multiple, spaced-out history frames** rather than only t−1. Owner's idea `[USER]`: helps understand movement, not just pose.
  - Detail: **exponential spacing** (≈ t−1, t−2, t−4, t−8, t−16), like dilated causal convolutions. **Space by time, not frame count** (timestamped ring buffer, sample/interpolate at fixed time offsets) since webcams drop frames and parts may run async. Feed **explicit differences** between samples, not only raw poses. `[CLAUDE]`
- **Lag discussion** (important nuance):
  - Owner pointed out: causal history adds **no pipeline latency** (only past frames). `[USER]`, correct.
  - Owner pointed out: training against the *current* (not smoothed) pose never rewards lag. `[USER]`, correct.
  - Claude's refinement: lag still appears as **hedging under uncertainty**. With MSE, the optimal output is the expected pose; at a sudden jump the model can't yet tell real onset from glitch, so it moves partway, then catches up 1–2 frames later. Also damps fast, brief motions. Longer history strengthens the motion prior → more hedging when prior and observation disagree. `[CLAUDE]`
  - `[CORRECTED]` Claude earlier implied longer windows directly add lag; refined to the hedging explanation above.
  - Levers: better evidence (less noise, backbone features, context), training statistics matching real use (plenty of sharp starts/stops, realistic noise), loss shaping. `[CLAUDE]`

### 3.7 Punishing over-hedging
Owner asked whether over-hedging can be penalized. `[USER]` (question). Options `[CLAUDE]`, mildest to strongest:
1. **Upweight onset frames** (detected in ground truth).
2. **L1 / Huber** instead of MSE (median instead of mean → commits when one explanation dominates).
3. **Asymmetric amplitude penalty** (undershoot costs more than overshoot).
4. **Adversarial motion loss** (discriminator punishes sluggish blends).
5. **Multi-hypothesis output** (pick most likely at runtime; needs hysteresis against pops).

Tradeoff: hedging also protects against glitches; less hedging → more committed false twitches. Mitigate with **realistic synthetic glitches** and by **measuring both sides** (onset lag/peak-velocity ratio *and* jitter/false-twitch). `[CLAUDE]`

### 3.8 Tunable behavior knobs
- **Knobs as model inputs that condition the loss weighting**: owner's idea `[USER]` ("a dynamic vs smooth parameter in the input vector would affect how the loss is weighed").
- Technique: **loss-conditional training** ("You Only Train Once", Dosovitskiy & Djolonga 2020). Sample knob per sequence, feed as input, use it to set loss weights. `[CLAUDE]`
- Implementation details `[CLAUDE]`:
  - Inject via **FiLM** (scale/shift activations), not plain concatenation.
  - **Per-part knobs** (e.g. snappy hands, calm torso).
  - Sample on a **log scale** of loss-weight ratio; oversample extremes.
  - **Change knob mid-sequence** sometimes in training so live slider moves don't pop.
  - **Verify monotonicity** by sweeping on held-out data; plot onset lag vs jitter.
- Candidate knobs `[CLAUDE]`:
  - Trained tradeoffs: **responsiveness vs smoothness**, **prediction horizon** (target t+Δ to cancel camera/render latency; overshoots at stops), **prior strength / detector trust**, **missing-part behavior** (hold / ease to rest / prior-generated).
  - Context conditioning (facts, not preferences): **seated vs standing / framing**, **avatar proportions** (preserve hand-face contacts across retargeting), **input frame rate / per-part update rates**.
  - **Not knobs** (do as post-processing): amplitude exaggeration, per-joint gain, deadzones, rotation limits.
- Suggested v1 set: responsiveness + prediction horizon, seated/standing as context. `[CLAUDE]` → `[OPEN]`
- Costs of multiple knobs `[CLAUDE]`: runtime negligible; real costs are capacity, training coverage, evaluation combinatorics, entanglement between overlapping knobs, UX (ship presets like Streaming/Gaming/Expressive, raw sliders in advanced panel). Add knobs only when plots show spare capacity.

### 3.9 Root translation and camera framing
- **Camera framing is not consistent** (angle, FOV, distance); training data must account for it. `[USER]`
- **Hip/root position must be model input and output** to capture translation; owner proposed normalizing to a null vector at t−1 (i.e. predicting deltas) to keep values bounded. `[USER]`
- **Hip position output now, with legs** `[USER]` (2026-11-14, pulled forward from M5; plan `dev-notes/plans/legs-root.md`): the body model and the merger output leg rotations (thigh, shin, foot) and the hip position in the **camera frame, metres, absolute per frame** (supersedes the delta idea above for now), plus an **up direction** (gravity in the camera frame) so the floor frame is one rotation of the whole pose away. Sitting vs standing is learned from the input, not a context switch; desk use is primary but other movement must keep working; the consumer may mask the lower body if it is unstable at the desk.
- Claude's refinements `[CLAUDE]`:
  - Deltas drift when integrated (random walk). Pair with a **slowly corrected absolute anchor**, complementary-filter style.
  - Express lateral/vertical translation in **body units** (shoulder width / torso length), not meters.
  - Depth: absolute is barely observable; **relative depth change** is (apparent size change) → predict as a **log-scale ratio**.
  - **Camera tilt vs body lean ambiguity**: add a short **calibration** step (neutral pose ≈1 s) as reference; output root position/orientation relative to it; feed calibration as context. Recalibration handles moved cameras.

### 3.10 Canonical skeleton and output
- **MHR is the canonical skeleton; VRM poses are derived from MHR rotations.** `[USER]` (2026-11-01; was `[USER-LEAN]`). The M3 spike still has to confirm the mapping is clean.
- **Two identities: performer and avatar.** The video shows the performer with (usually) realistic proportions; the avatar has its own, often stylized ones. Both are MHR identities and must be kept distinct: the pose is tracked on the performer's proportions and then mapped onto the avatar's. `[USER]` (2026-11-01)
- **Target avatars' proportions are modeled as MHR identities**: each VRM avatar gets an MHR fit, and the retargeting layer (§3.10a) maps the performer's pose onto that fit. `[USER]` (2026-11-01). Owner will provide a few VRM models for this. Open risk: whether MHR's human-derived identity space can represent stylized (VRoid) proportions closely enough; see the M3 spike.
- Claude's proposal: thin fixed retargeting layer from MHR to **VRM 1.0** at the output, **OpenXR hand joints** as an optional second output. Fallback: own canonical skeleton that's a superset of VRM bones, with MHR used only for rendering. `[CLAUDE]`
- Reasons for MHR: Apache 2.0 (avoids SMPL licensing), explicit bone lengths (fits avatar-proportion conditioning), body+hands+face in one model, provides licensed realistic meshes for rendering, aligned with where the ecosystem is going. `[CLAUDE]`
- **M3 spike findings (2026-11-01)**, details in `dev-notes/plans/m3-mhr-vrm-spike.md`: rotations map onto VRM 1.0 with a fixed per-bone rest offset; bone directions exact, end-effector error under ~1 cm from folding joints VRM lacks. `[CORRECTED]` Bone lengths come from MHR's additive `scale_*` model parameters, not the 45 identity coefficients (those shape the mesh only). Stylized VRoid shoulders and some lower legs fall outside MHR's human limits (soft penalties, still representable). MHR's `*_flexible` joint translations can't be expressed in VRM.
- **M3 spike result (2026-11-01): go.** CMU → MHR fitting works (median residual ~1 cm, limbs below that). Plain rotation transfer onto stylized VRoid avatars misses close contacts by ~1–5 cm median, up to ~7 cm (p90), while preserving them exactly on an avatar with the performer's proportions: proportion differences, not the conversion, cause the misses. Confirms the need for the retargeting layer (§3.10a).
- **Canonical skeleton rules from the M3 spike** `[USER]` (2026-11-01):
  - Avatar MHR fits may go beyond MHR's human scale limits; the performer identity stays within them.
  - MHR's `*_flexible` joint translations stay at zero in the canonical pose (VRM bones can't translate).
  - Forearm twist goes onto VRM `lowerArm` via a swing-twist split, not onto the hand.
    - `[CORRECTED]` (2026-11-16): until 2026-11-16 the mapping carried MHR's rest offset. MHR's `wrist_twist` and `lowarm` rest frames differ by ~114 deg about the forearm, so `lowerArm` carried ~130 deg of twist at rest and the hand undid it (forearm wrung at the elbow and back at the wrist; hand world rotation right). Present since the first conversion code (2026-11-01: `m1-v10` and all runtime output through v0.1.0). Fixed in v0.1.1: only the twist change since rest counts. Details: `dev-notes/modules/mtcore.md` (convert).
  - **A wrist can't roll**: after the mapping, any twist about the forearm left in the VRM hand's rotation relative to `lowerArm` moves onto `lowerArm`; hand world rotation unchanged, the hand's local rotation keeps only swing. Spreading the roll onto the upper arm is not ours: that is a renderer concern vspark already handles (forearm twist bones, `forceTwistBone`). `[USER]` (2026-11-16)
- Known risk: MHR's pose parameterization is richer and less game-engine-shaped; mapping trouble spots likely: **spine segment count, forearm twist joints (VRM has none; fold twist into lower arm/hand), finger/thumb conventions, rest pose**. `[CLAUDE]` → needs spike (§9)
- **Gaze**: owner sees no reason to treat it differently from other outputs. `[USER]`
  - Detail: since VRM look-at is bone- or expression-based per avatar, output **canonical gaze angles per eye** and let the adapter map to bones or expressions (three-vrm's look-at handles both). `[CLAUDE]`

### 3.10a Learned retargeting layer
- **A learned retargeting layer maps the performer's pose onto the avatar.** Avatar bone lengths are fed only into this layer; everything before it works in the performer's proportions and stays avatar-independent. It is trained explicitly to keep spatial accuracy (relative positions and directions between body parts, contacts included) and can read earlier latents so temporal history informs the targeting. `[USER]` (2026-11-01)
- Rejected alternatives (2026-11-01): a purely analytic IK step that only snaps explicit contacts (sees one pose, flickers near contact thresholds, misses pre-contact geometry); explicit contact probabilities from the model plus analytic IK (too narrow: only contacts). `[DISCARDED]`
- Details proposed by Claude, not yet confirmed `[CLAUDE]`:
  - **Placement after the merger**, since the relations that matter span parts (left finger to right finger, hand to face).
  - **Residual on plain rotation transfer**: output = performer rotations copied to the avatar + learned correction. Worst case degrades to ordinary retargeting; identity when proportions match.
  - **Explicit pose + confidence as main input, latents optional with dropout**, so head retrains don't force a retargeting retrain (same rule as §3.4).
  - **Self-supervised objective**: preserve vectors between key points (fingertips, hands, face, chest) scaled to each body, weighted up as pairs get closer; regularized by rotation similarity, joint limits (also serves the no-impossible-poses requirement, §3.3) and smoothness. Training pairs = any motion × randomly sampled avatar MHR identity; no paired labels needed. Prior art: spatial-relationship-preserving retargeting.
  - **Evaluation split**: performer-space hand accuracy label-free on real video; retargeting quality on synthetic poses with the same relation metric.
  - **Timing**: not needed for M1; fits M4, after the M3 MHR mapping spike.
- **Runtime retargeting, first live step (2026-11-16)** `[USER]`: the relational retarget prototype (`mtcore.relretarget`, analytic, relation-preserving, not contact snapping) goes into the runtime as a per-avatar stage after the merger, to see live whether it helps before anything learned. The human side is MHR's mean body (the performer's proportions are unknown).
  - Retargeted output only fits the avatar it was computed for, and one input may drive several avatars (several in vspark, or a VMC receiver with its own avatar). Design: the default VMC stream is retargeted to the avatar given as a VRM (`--avatar-vrm`, the `avatar` command's `vrm_path`); with no VRM it uses the default reference rest pose and skips retargeting. Later: a command that adds an output (avatar plus VMC destination host:port, one destination port per avatar since VMC carries no avatar id) and returns an id for removing it. `[USER]`
  - Avatar data: the VRM file path for now (shared filesystem); a compact avatar profile (rest positions plus surface measurements) later, so remote clients can register avatars. `[USER]`
  - ~~`--no-retarget` switches it off~~ `[CORRECTED]` 2026-11-17, see below.
  - Measured on renders with Fenn_2 (2026-11-16): with the true pose it fixes contacts (palms together 0.1 cm vs 1.0 for rotation copy, hand to face 0.5 vs 1.4, belly 0.3 vs 2.3). On tracked input it only partly helps: separated hands improve 2-3 cm (rotation copy puts them too close), touching hands get worse (palms together 5.6 vs 1.4 cm: the tracked pose has the palms 13 cm apart on MHR's mean body and retargeting carries that over faithfully, while copying onto the smaller avatar hid it), and hand-to-face / hand-on-body stay as off as before (21 cm: tracking misses, beyond the falloff). The same renders reproduce the owner's "hands at face level shown at chest level": in hand-to-face segments the model puts the wrist about 10 cm low.
  - A toggle, off by default, switchable live from vspark (stdin `{"cmd": "retarget", "enabled": bool}`, answered by a `retarget` event; `--retarget` at start), so the owner can compare on and off directly. `[USER]` 2026-11-17 (default off: `[CLAUDE]`, following the measurement)
  - Order: the default stream first, measured on renders with Fenn_2 and compared live by the owner; the add-output command only if it helps. `[CLAUDE]` (proposed, owner did not object)

### 3.11 Blendshapes
- Not the owner's focus (many setups use ARKit), but **include for completeness**. `[USER]`
- Owner suggested expressions might be **useful history/context** for body motion (smiling vs frowning moves differently), while noting it's hard to build into training data. `[USER]` (idea)
- Claude's proposals `[CLAUDE]`:
  - Face head outputs **ARKit 52 blendshapes**; adapter maps to Perfect Sync avatars directly, otherwise collapses to VRM presets.
  - Start the face head as a **temporal refiner on top of MediaPipe's blendshape model** (fine-tune from Google's weights rather than train from scratch).
  - Labels: **rendered Perfect Sync VRoid avatars** with random blendshape sequences (exact labels); and **ARKit-as-teacher self-recordings** (iPhone + webcam simultaneously → ARKit blendshape + head-pose labels for webcam frames, no license issue).
  - Expression-as-context: add blendshapes to the shared context channel **now, with masking**, so architecture supports it; learn the correlation later from real paired recordings (ARKit+webcam sessions naturally contain face–body co-occurrence). Synthetic data with independent expressions would teach "unrelated" (harmless, useless).
- **First slice (2026-11-12): MediaPipe's own blendshape model on Holistic's face landmarks** in the runtime, no learned face head yet. `[USER]` Revisit later: the merger may want blendshapes (or the face model's features) as input. `[USER]` See §7.5.
- **Algorithmic post-processing first (2026-11-14):** the runtime corrects MediaPipe's blendshapes algorithmically (eye resting level, look-down and left/right correction, 60 Hz glide; §7.5); the learned face head (the `[CLAUDE]` proposals above) stays in reserve. `[USER]` (2026-11-14) Motivation (measured): open eyes rest at eyeBlink 0.25-0.41, rising with eyeLookDown (correlation ~0.7), eyeSquint rests at 0.34-0.45 and vspark's ARKit → VRoid table adds 0.3 × squint to eye closure, so eyes looked ~40% closed; left and right differ noticeably.
- **MediaPipe face cost**: owner remembered the blendshape variant as much heavier. Claude's explanation: blendshape stage is <1 ms; heaviness was likely plumbing: running legacy Face Mesh *and* Face Landmarker (double detection), IMAGE running mode (no tracking), or GPU delegate overhead on a tiny model. `[CLAUDE]` → worth re-checking in baseline.

### 3.12 Out of scope
- **Person selection** is a base-model/detector problem, not this project's. `[USER]`
  - Caveat `[CLAUDE]`: once MediaPipe plumbing is reimplemented for feature taps, we own detect/track, including **associating hands and face with the same person as the body**.

---

## 4. Expected compute `[CLAUDE]`
- Each part head: small temporal MLP / dilated causal conv, ≈0.1–0.5M params.
- Merger: similar or smaller (residuals).
- Total: well under 2M params, a few million ops/frame, ≈0.1 ms on a modern CPU. Small fraction of MediaPipe's cost.
- Ring buffers: negligible memory.
- **Call overhead dominates at this size**: export all heads + merger as **one ONNX graph**, run once per frame, **on CPU** (don't compete with renderer/game on GPU).
- Training: hours to ~a day per run on one consumer GPU. Offline data generation (rendering) is the heavy part.

---

## 5. Data strategy

### 5.1 Motion sources
- Must be permissively licensed. `[USER]`
- **CMU approved and downloaded** as primary real mocap. `[USER]` (2026-11-01). Original ASF/AMC from mocap.cs.cmu.edu into gitignored `data/` (never commit: the terms forbid reselling the data, even converted). UPenn as second source and 100STYLE (license unconfirmed) remain `[CLAUDE]`.
- Retarget everything to the canonical skeleton yourself; never via SMPL. Randomize bone lengths. `[CLAUDE]`
- Budget time to **clean CMU** data (varying skeletons per subject, noisy clips). `[CLAUDE]`

### 5.2 Seated / upper-body gap
- Owner wants **full-body data included too**, but **lack of seated data is a real gap**. `[USER]`
- Owner proposed **synthetic data** to fill it, noting the end of the stack doesn't care about realistic video, and that artificial movement might still be enough for a **proof of concept**. `[USER]`
- Claude agreed: lifter never sees pixels, so image realism doesn't matter for it; **motion realism** does (temporal model learns its motion prior from training data). Techniques `[CLAUDE]`:
  - **Splice** real upper-body motion (arms/neck/head from CMU gesture/reach/talk-like clips) onto a seated lower body; add procedural leaning (seated people lean/shift, don't transfer weight between feet).
  - **Procedural motion from human movement models**: minimum-jerk reaching trajectories (Flash & Hogan; bell-shaped velocity profiles) to desk-relevant targets (keyboard, mouse, face, hair, cup, chest gesture space); joint coupling (scapula moving with arm raise); **idle life** (breathing, sway, small head motion, fidgets).
  - **Unlabeled real seated video** of the owner at a desk → self-supervised reprojection loss (§5.4) closes the remaining gap.
  - For the POC: keep history window short initially so a slightly wrong motion prior can't dominate.

### 5.3 Rendering pipeline ("render, then detect")
- Owner's idea: render poses to images to pad training data and get the most accurate pose labels. `[USER]`
- Claude's key addition: **render → run frozen MediaPipe on the render → real detector output paired with perfect ground truth.** Replaces most of the hand-built noise model with MediaPipe's real failure modes (hand swaps, occlusion errors). `[CLAUDE]`
- Characters: **VRoid** (stylized; own your exports) + **realistic humans** (MHR meshes Apache 2.0; MakeHuman CC0, verify) so the model doesn't learn anime proportions/shading as normal. `[CLAUDE]`
- Webcam realism: motion blur, low light, sensor noise, compression, varied camera angles/FOV/backgrounds, realistic **desk-webcam framing**. `[CLAUDE]`
- Rendering engine: owner **leans three.js** (existing stack), open to Blender if easier. `[USER-LEAN]`
  - Claude: three.js for now (POC lifter needs no images; three-vrm handles VRoid natively). Move to **Blender** when render realism becomes the bottleneck (headless batch rendering, imports MHR FBX rigs, has a VRM add-on). `[CLAUDE]` → `[OPEN]`

### 5.4 Learning from unlabeled video
- Owner asked about **reinforcement/adversarial** routes not relying on training data, and whether limited unlabeled video is more than a benchmark. `[USER]` (question)
- Claude's answer `[CLAUDE]`:
  - Unlabeled video is useful beyond benchmarking: (1) **calibrate the noise model** (jitter spectrum, dropout rates, failure modes from people holding still or moving slowly), (2) **self-supervised fine-tuning** via **reprojection loss** (FK predicted rotations → project to 2D → compare with detected keypoints) plus plausibility terms (joint limits, smoothness).
  - **Adversarial motion prior** (discriminator trained on permissive mocap judges whether output motion looks human). Needs no paired data. Good fit.
  - **RL**: poor fit (see §8).

### 5.5 Synthetic noise model `[CLAUDE]`
Frequency-shaped jitter; depth noise ≫ 2D noise; dropouts; confidence-correlated error; occasional left/right hand swaps; sudden glitches. Calibrate against real MediaPipe output (§5.4). Realistic noise matters more than which anti-hedging loss is chosen.

### 5.6 Hands `[CLAUDE]` → `[USER]` (2026-11-12)
Hands have the least permissively licensed mocap → **procedural finger poses** within anatomical joint limits, smooth random trajectories between them. `[USER]` (2026-11-12)
- Hand model v2 (owner, 2026-11-12, after the h2 comparison video): `[USER]` unless marked
  - The hand model also outputs the hand's **absolute rotation and position** (camera frame), so it stays independent of the body; the wrist bend is computed downstream from the body's forearm (merger later, a simple combination step first). `[USER]`
  - **Position and rotation history as input** (helps tell MediaPipe's palm/back flips from real turns); remaining ambiguity to the merger. `[USER]`
  - **Hands as a pair**: shared per-hand encoder (left mirrored) plus a joint layer seeing both hands, so swaps, occlusion and hand-to-hand contact can be learned; each hand still works alone. `[USER]` Cross-part relations (hand-face, retargeting) stay with the merger (§3.10a). `[CLAUDE]`
  - **Explicit hand interactions, temporary flips and hand swaps in the training data**: interactions as real two-hand motion (IK choreography), flips and swaps as detector failures calibrated on renders of those interactions through real MediaPipe. `[USER]`
  - Combination step puts the hand's twist about the forearm onto the lower arm (overriding the body model's), only the bend onto the wrist, with wrist limits and flip rejection. `[CORRECTED]` (2026-11-13): the rendered truth carries the forearm twist on the hand bone, and moving it onto the lower arm put the lower arm 129 deg off; the combination step now puts the whole rotation on the hand, rejecting implausible bends. `[CLAUDE]` `[CORRECTED]` (2026-11-16): the 129 deg came from the `MhrToVrm` forearm twist bug (fixed in v0.1.1, 43347af: wrist_twist's rest frame was taken as twist), not from the truth convention. With the render ground truth regenerated, `twist_to_forearm` is on by default (eee0b04), following the owner's wrist-roll rule `[USER]` (§3.10): twist on the lower arm, only swing on the hand. p3 on pairs2 (Holistic), median deg: lowerArm world 58.6 (twist on hand / m1-v10: 53.2), hand world 53.9 (m1-v10: 64.3), hand local 29.7 (twist on hand: 39.4; m1-v10: 30.8). Every system is 50-65 deg off in world orientation; unexplained `[OPEN]`.
- Hand model v3 (owner, 2026-11-13, after the p2 comparison video: counting still weak although the landmarks follow; lost hands keep their last pose instead of relaxing; hands near the body clip a little): attack it in the training data. `[USER]`
  - **More independent finger poses** (fingers extended/curled independently, counting in more styles). `[USER]`
  - **Explicit off-screen sections**: hands leave the frame and settle into a varied relaxed rest (fingers and wrist). The model learns the probable pose and orientation while unseen (a hand leaving at the bottom probably hangs) for fingers **and** rotation, and **emits confidence**; the merger prefers the body's pose input when the hand's confidence is low. `[USER]`
  - The model also **ingests a detector confidence**: one generic per-hand score slot (Holistic: the pose model's visibility of the wrist/hand points; Hand Landmarker: its handedness score), synthesized from each detector's calibrated distribution, so training stays flag-free. `[USER]`
  - **Body contact poses** (hands in the lap, on thighs/belly/chest, crossed arms), targets collision-free where the landmarks would clip. Data only; body landmarks as input may come with the merger later. `[USER]`
  - Correction data (observed hand landmarks clipping into the body, target pose corrected outward) belongs to the merger, which sees body and hands; the hand model doesn't see the body. `[USER]` (2026-11-13)
  - **Hand detector** (2026-11-12): MediaPipe's standalone Hand Landmarker (own palm detector) finds hands without the head in view, where Holistic finds almost none (owner's first hands-v1 take: Holistic 0-19%, Hand Landmarker 80-100%). On renders (head in view): fewer hands detected (52% vs 76% of visible hands; small hands) but far fewer palm/back flips (0.6% vs 9.7%) and fewer swaps with the hands close (6.9% vs 19%); same shape error. Owner leans toward offering it (hands-only users with headless framing), possibly as the default. `[USER-LEAN]` Runtime setting holistic / hands / both. `[CLAUDE]` The hand model is trained on a mix of both detectors' error patterns, with no detector flag, so it stays detector-agnostic while it only reads landmarks; once it reads a detector's internal features (latents) it is tied to that detector. `[USER]` (2026-11-12) A flag with dropout can be added later if specializing helps. `[CLAUDE]` What drives the arms in hands-only mode (no body): decide later. `[USER]`
  - Order: wrist first, then the finger improvements (independent fingers and counting, bend axis) from the owner's feedback on h2. `[USER]`
- First learned hand model (plan `dev-notes/plans/m4-hand-model.md`): synthetic MediaPipe hand landmarks from MHR FK plus a noise model calibrated on rendered hands `[USER]`; input = the hand's own landmarks only for now, deferring full-pose context (§3) `[USER]`; judged on rendered hands with exact ground truth against the ported solve `[USER]` (all 2026-11-12).

### 5.7 Optional real paired data `[CLAUDE]` → `[OPEN]`
- **Consumer IMU suit** (e.g. Sony mocopi, a few hundred dollars): record yourself with suit + webcam simultaneously → paired real data with rotation labels in the exact target domain. IMU drift is manageable over short clips. Possibly the highest-value purchase in the project.
- **ARKit + webcam** simultaneous sessions for face/head labels (§3.11).
- Note: owner said they can't realistically provide their own training data; these are optional, cheap options that may change that.

---

## 6. Evaluation `[CLAUDE]` (baseline extraction is `[USER]`)
- **Build the harness before any model.** Measure the **current MediaPipe pipeline first** (owner will extract it as baseline `[USER]`).
- Metrics:
  - Rotation error (where labels exist)
  - **Onset lag** (cross-correlate predicted vs ground-truth joint trajectories)
  - **Peak-velocity ratio** (damping of fast motion)
  - **Jitter at rest** / false-twitch rate on still and slow segments
  - Contact preservation (hand-face, hands-together)
  - Reprojection error on unlabeled real video
- Test sets: held-out synthetic, ARKit-labeled face/head recordings, unlabeled real seated video.
- **Success criteria for the M1 go/no-go**, provisional `[USER]` (2026-11-01: "confirm it for now to get going"); revisit after more seated-v2 sessions. The model is measured as a replacement for both vspark's solver and its frontend One Euro filter: its output is upsampled to 60 Hz without smoothing (glide to the latest pose; harness `--system <name>-glide=<poses>`) and compared against `baseline-rendered`, what the avatar shows today `[USER]` (2026-11-08; was: both through the same render filter). The model through the render filter (`<name>-rendered`) is reported as a secondary number. On at least three seated-v2 sessions, holding on each session. Body bones only (shoulders, arms); hand/finger targets belong to M4.
  1. Rest noise with arms below the frame: at least 50% lower than baseline.
  2. Twitches per minute at rest: under 10 (baseline ~38–49).
  3. Rest noise with elbows in view: no worse than baseline.
  4. Added lag vs detector: no more than baseline, ideally under 50 ms (harness resolution is coarse at 10 FPS). Note: the harness stamps speeds at frame-interval midpoints, so any causal 60 Hz display of a 10 FPS stream shows at least half an interval (~50 ms); glide measures ~90 ms, about the same as `baseline-rendered` (70–95 ms).
  5. Speed p90 in slow/fast segments: at least 80% of baseline (guards against over-smoothing).
  6. Model cost: under 2 ms per frame on a laptop CPU.
  7. Hand position `[USER]` (restated 2026-11-09; was: 3D error vs detected hands at least 30% lower than baseline, which favoured the baseline: it copies the very landmarks the metric compares against, including MediaPipe's unreliable depth). (a) Wrist error on the performer's own skeleton in the camera plane, against the image landmarks: no worse than baseline; guarded by the forearm direction in the image (errors along the viewing axis barely move the wrist in 2D). (b) Contact: during the seated-v2 contact segments (palms together, hand to face; true distance about 0), the avatar's distance on the performer's skeleton is below baseline. Threshold to set once sessions exist.
  8. Joint-limit violations: near zero in all conditions, including low confidence and tracking loss.
  9. Side-by-side video check by the owner: at least as natural, especially with arms below the frame.

---

## 7. Service architecture

### 7.1 Shape
- **Standalone service first**, connected via WebSocket/UDP/etc; **language-specific libraries later** for lower latency. `[USER]`
- **Service owns the camera** and sends only poses (a few hundred bytes/frame) by default; shipping frames between processes is where cost goes. `[USER]` (2026-11-12: camera, MediaPipe and the pose model in one runtime, since feature taps need direct access to the MediaPipe models)
- **Also support an incoming media stream** (owner's idea), e.g. a **browser client** for testing the service without the full consumer app. `[USER]`
  - Frame-source abstraction: local camera, video file (repeatable tests), remote stream. `[CLAUDE]`
  - Remote stream options: **JPEG over WebSocket** (simplest, fine for test client) → **WebRTC** later (proper real-time; enables phone-as-camera). `[CLAUDE]`
  - Compressed input adds encode/decode latency and shifts input distribution (training includes compression artifacts). For remote sources, **capture timestamp comes from the client and is echoed back**. `[CLAUDE]`
- Browser client also as a **pure viewer** when service owns the camera: optional low-rate preview with keypoint overlays + avatar via three-vrm. Debug dashboard and later demo. `[CLAUDE]`

### 7.2 Protocol
- **WebSocket mirrors the VMC message shape**: owner's idea `[USER]`.
  - Detail: OSC is transport-agnostic; send the same binary OSC packets as WebSocket binary frames (e.g. osc-js supports UDP + WebSocket, Node + browser). One encoder, one schema. `[CLAUDE]`
  - **VMC over UDP** as an additional sink for ecosystem compatibility. `[CLAUDE]`
  - Extensions `[CLAUDE]`:
    - **Capture time in OSC bundle timetags.**
    - Confidence, presence flags, sequence numbers, calibration state under **own address namespace** (VMC receivers ignore unknown addresses). Namespace name: `[OPEN]`.
    - **Control messages** (knobs, calibration, part toggles, camera selection) back over the same WebSocket, also OSC.
    - **Version the schema** from day one.
  - Gotchas: VMC sends bone names as strings (verbose, irrelevant on localhost); **Unity left-handed vs three.js/glTF right-handed** → clients need conversion. `[CLAUDE]`
- Delivery semantics: **latest-wins**, never queue stale poses. `[CLAUDE]`
- Local transport latency is sub-millisecond; camera exposure/buffering and inference dominate. `[CLAUDE]`

### 7.3 Implementation language
- **POC service in Python** (ONNX Runtime, MediaPipe), with an interface matching what the native core will expose. `[USER]` (2026-11-12)
- Later: **native core library (C++ or Rust)**, service as thin wrapper; language libs = bindings (N-API for Node, Python, C# for Unity). `[CLAUDE]` → language choice `[OPEN]`
- Feature taps require **leaving the MediaPipe Tasks API** (it doesn't expose intermediate tensors): run the models directly and **reimplement detect-then-track plumbing** (pose ROI from previous frame's auxiliary landmarks 33/34, etc). Real engineering work; plan the keypoint-only lifter as a complete milestone first. `[CLAUDE]`

### 7.4 vspark integration (2026-11-12)
Pulled forward from M2 so the model can be tried live. Plan: `dev-notes/plans/m2-vspark-runtime.md`.
- **vspark gets its own behavior** (kind `learned_tracker`, label "Learned Tracking") so the runtime feels native. `[USER]`
- **Runtime loaded lazily from this repo, not bundled into vspark**: vspark starts it through `uv` from a local checkout now, from the public git URL later. `[USER]` The "public git URL later" part (`uvx --from git+...`) is superseded (2026-11-14) by the release asset download below; that route installs the code only, without models.
- **vspark downloads the runtime release asset itself** (`motiontracker-runtime-<version>.zip`: `core/` + `service/` at the tag plus model bundles, run with `uv run --project motiontracker-<version>/service mtservice`). The version is pinned by vspark (`packages/backend/src/behaviors/learned_tracker/download.ts`, `MOTIONTRACKER_RUNTIME`) and updated together with vspark. `[USER]` (2026-11-14) First release v0.1.0 (2026-11-14).
- **Output: VMC** (OSC over localhost UDP); confidence and presence under an own OSC namespace. `[USER]` ("VMC if that's the most suitable format")
- **Control over the child process's stdin/stdout** (JSON lines: start/stop, camera, avatar; status and camera list back). WebSocket deferred to the browser viewer. `[USER]` (deviates for now from §7.2's WebSocket-first shape)
- **Avatar rest pose passed by vspark at start** (limb rest directions for MHR → VRM), VRoid rest as the default. `[USER]` All CC0 VRoid samples share the same rest offsets from an ideal T-pose (upper arm 2.7, forearm 4.7, hand ~8.3 deg).
- **The model drives every bone it predicts**, head included; vspark's mixer combines it with other sources. `[USER]`
- **First model: m1-v10.** `[USER]`
- Weights stay in this repo's gitignored `data/`; vspark never holds them. `[CLAUDE]` Superseded 2026-11-14: weights are published as release assets (§9 #9) and vspark downloads them inside the runtime asset.
- Repo made public 2026-11-12 without a license yet (all rights reserved). Weights to be published separately later (release asset with model card), never CMU-derived data. `[USER]` Published from v0.1.0 (2026-11-14, §9 #9).

### 7.5 Face and hands in the runtime (2026-11-12)
Plan: `dev-notes/plans/m2-face-hands.md`. Model stays m1-v10 for the body. `[USER]`
- **Face:** MediaPipe's blendshape model (from the Face Landmarker bundle) run on the face landmarks Holistic already produces, 52 ARKit values out over VMC; no second face detector (vspark runs Holistic + Face Landmarker today, detecting the face twice). `[USER]` To revisit when the merger is built (blendshapes as merger input). `[USER]`
  - `[CORRECTED]` (2026-11-12) The Holistic bundle ships the same blendshape model (byte-identical) and Holistic's Python API outputs it (`output_face_blendshapes`); vspark's note that it is a no-op holds for WASM at most. So no extra interpreter is needed.
  - Face mesh: Holistic ships the *lite* Face Mesh V2, Face Landmarker the *full* one (same 478 points). Swapping the full mesh into the Holistic bundle raises blendshape agreement with Face Landmarker from median correlation 0.72 to 0.87 on the session (runtime, 662 frames) for about 0.2 ms/frame; body output bit-identical. Runtime flag `--face-mesh full|lite`, default `full`; built locally from the two pinned Google downloads, not redistributed. Flag only, no vspark setting for now. `[USER]` (2026-11-12)
  - When face-model features (latents, M7) feed learned parts, lite and full stop being interchangeable: their internal features differ. `full` is then the reference the learned parts train on; `lite` remains a cheaper mode without learned face features (merger trained with features dropped out, §5). `[CLAUDE]`
  - Post-processing (2026-11-14, §3.11): eye correction (`mtcore.face_filter`) and a glide to 60 Hz in the runtime, configurable with `--face-rate` (0 = send per detector frame, for receivers that interpolate themselves). `[USER]` (2026-11-14) Whether vspark should also interpolate blendshapes on its side: `[OPEN]`.
  - Left/right: the avatar mirrors the person. On the mirrored frames MediaPipe's pose "left" is the person's right, and its *Left blendshapes follow the image's right side as well (measured), so face and body agree when the blendshapes are passed on unswapped, as vspark's MediaPipe Tracking does. `[CLAUDE]` `[CORRECTED]` (2026-11-12): a first version swapped them to the person's own left, which put winks on the opposite side from the arms. Note: vspark's VMC "mirror" option swaps bones only, not ARKit shapes.
- **Hands:** port vspark's algorithmic finger solve into the runtime first so fingers work end to end, then replace it with a learned hand model (procedural finger poses, §5.6). `[USER]`
- Same runtime, same VMC stream (blendshapes as `/VMC/Ext/Blend/Val` with ARKit names, fingers as bones). `[CLAUDE]` vspark's VMC receiver reads ARKit only from RhyLive's `/Face` today, so it gets standard `Blend/Val` support. `[CLAUDE]`

### 7.6 Merger in the runtime (2026-11-14)
Branch `feature/merger-runtime`. Merger decisions: §3.4; legs/hip/up: §3.9.
- **Merger `mg3` goes live first** (legs, hip position as root in the camera frame, up direction; body model `m1-v14`). `[USER]` (2026-11-14)
- **ONNX + numpy runtime, no torch or pymomentum in the runtime.** `[USER]` (2026-11-14) Separate ONNX exports of `m1-v14` (root model), the `p3` pair hand model and the merger; numpy ports of the code in between (hand model frame features, face-mesh head pose already numpy in `mtcore.facepose`, the base combination `mttrain.merger.batch.combine`, finger parameters → VRM finger locals via an exported table). Note: three ONNX graphs plus numpy glue, not §4's single-graph idea (`[CLAUDE]`, unchanged).
- **Leg bones on by default**: vspark's mixer can zero the leg influence; `--no-legs` drops them for other consumers. `[USER]` (2026-11-14)
- Hip position and up direction go out as VMC extensions `/MT/Root x y z` (m) and `/MT/Up x y z` (unit), in the body model's camera frame (x right in the mirrored image, y down, z away), not vspark's pose space; `Root/Pos` stays zero; vspark ignores both for now. `[CLAUDE]` (2026-11-14, following the owner's answer)

---

## 8. Discarded and deprioritized approaches

| Approach | Status | Reason |
|---|---|---|
| Pure algorithmic MediaPipe pipeline (One Euro filtering, slerp smoothing, confidence weighting, Kalidokit-style solvers) | `[DISCARDED]` by `[USER]` | Already exhausted, results unsatisfying. Remains the **baseline**. |
| **RTMW / RTMW3D** whole-body | `[DISCARDED]` `[CLAUDE]` | ~65M params (L), ~30 fps on 3090; weights trained on mixed-license data. |
| **YOLO pose** (Ultralytics) | `[DISCARDED]` `[CLAUDE]` | AGPL. |
| **MoveNet** | `[DISCARDED]` `[CLAUDE]` | Only 17 keypoints, no training code. |
| **ViTPose / Sapiens** | `[DISCARDED]` `[CLAUDE]` | Too heavy; Sapiens non-commercial. |
| **OpenPose** | `[DISCARDED]` | Non-commercial license. |
| **HaMeR / WiLoR** (hand mesh) | `[DISCARDED]` `[CLAUDE]` | Better hands but too heavy for target latency/hardware. |
| **SMPLer-X, PyMAF-X, WHAM, 4DHumans, MotionBERT, DanceHMR** etc. | `[DISCARDED]` `[CLAUDE]` | Offline/heavy; mostly SMPL-based (license). |
| **AMASS, SMPL/SMPL-X, LaFAN1, Bandai Namco mocap** | `[DISCARDED]` | Non-commercial; conflicts with `[USER]` licensing requirement. |
| **Pseudo-labeling video with heavy non-commercial models** (WHAM, HaMeR…) | `[DISCARDED]` `[CLAUDE]` | Labels inherit licensing risk. (SAM 3D Body: possible later, only if its license allows.) |
| **MocapNET / XR Animator as dependencies** | `[DISCARDED]` | Custom license / CC BY-NC-SA. Reference only. |
| **Reinforcement learning** (incl. physics-based trackers) | `[DISCARDED]` `[CLAUDE]` | Heavy, fragile, poor fit. Adversarial prior + reprojection used instead. |
| **Temporal input inside the detector backbone** | `[DISCARDED]` `[CLAUDE]` | Multiplies the expensive part. Temporal fusion in heads over cached features instead. |
| **LoRA on the backbone** | Deprioritized `[CLAUDE]` | Works on convnets, but a light full fine-tune is simpler once training code exists. LoRA only useful for keeping base weights shared/small. |
| **Latent-only head → merger interface** | `[DISCARDED]` as primary `[CLAUDE]` | Loses swappability, staged training, degradation. Small latent kept as optional addition with dropout. |
| **Single fed-back frame as the only temporal context** | `[DISCARDED]` (owner) | Replaced by spaced multi-frame history. |
| **gRPC / HTTP** for the pose stream | `[DISCARDED]` `[CLAUDE]` | Complexity without benefit locally. |
| **JSON** for the pose stream | `[DISCARDED]` `[CLAUDE]` | Binary OSC instead; JSON only ever considered for control. |
| **App captures frames and ships them to the service** as the primary mode | `[DISCARDED]` `[CLAUDE]` | Cost/latency. Remote frames still supported as a secondary source. |
| **Training amplitude exaggeration / gains / deadzones as knobs** | `[DISCARDED]` `[CLAUDE]` | Simple post-processing; no need to learn. |
| **100STYLE** | Deprioritized | License unconfirmed; locomotion-only. |
| **CMU FBX conversion on 4TU.ResearchData** | `[DISCARDED]` | Relicensed CC BY-NC. Use original/cgspeed BVH. |
| **Synthetic MediaPipe body co-rotation with the head** (m1-v11) | `[DISCARDED]` by `[USER]` (2026-11-12) | Better on renders, but on the real session the avatar slumped and looked down, and head turns looked better coming from the head in v10; nothing visibly improved. A follow-up with wider torso-tilt randomness was not trained: the session's MediaPipe torso pitch (median -14 deg) already sits mid-distribution of the synthetic data (median -16 deg). Plan: `dev-notes/plans/m1-v11-body-corot.md` (branch `feature/m1-v11-body-corot`). |

---

## 9. Open questions

| # | Question | Owner of decision | Notes |
|---|---|---|---|
| 1 | ~~**MHR → VRM 1.0 maps cleanly?**~~ Resolved 2026-11-01: yes (M3 spike, §3.10). | Owner (after spike) | Spike: load MHR rig, list joints, build mapping table; round-trip VRM→MHR→VRM on CMU clips; check spine, twist joints, fingers/thumb, rest pose. |
| 2 | ~~**MediaPipe vs RTMPose** as base~~ | Owner | Resolved 2026-11-01: MediaPipe `[USER]` (§3.2). |
| 3 | **three.js vs Blender** for rendering | Owner | three.js first is Claude's suggestion. |
| 4 | **Success criteria** numbers | Owner | Provisional set confirmed 2026-11-01 (§6); revisit with more sessions. |
| 5 | **OS targets** | Owner | Affects runtime choices (GPU support varies by platform). |
| 6 | **v1 knob set** | Owner | Suggested: responsiveness, prediction horizon, seated/standing context. |
| 7 | **100STYLE license** | — | Check Zenodo page if still wanted. |
| 8 | **MakeHuman asset license**, **MediaPipe blendshape weights license** | — | Verify against official sources. |
| 9 | **Own project license** (likely Apache 2.0 for code + weights, with NOTICE attributions for MediaPipe/MHR) | Owner | Owner wants permissive `[USER]`; exact license open. Weights may be published: live model bundles go out as GitHub release assets (`motiontracker-models-<version>.zip`, first v0.1.0) `[USER]` (2026-11-14); the repo has no LICENSE yet. |
| 10 | **IMU suit purchase / ARKit recording sessions** | Owner | Optional. |
| 11 | **Native core language** (C++ vs Rust) | Owner | After POC. |
| 12 | **Calibration UX** (when to prompt, camera-moved detection, recalibration feel) | Owner | |
| 13 | **OSC extension namespace** name | Owner | |
| 14 | **Cross-part person association** after reimplementing plumbing | — | §3.12 caveat. |
| 15 | **Consent/privacy** if ever collecting others' recordings or opt-in user data | Owner | |

---

## 10. Suggested milestones `[CLAUDE]` (order not confirmed)

**Order change `[USER]` (2026-11-01):** M3 (MHR ↔ VRM spike) runs before M1, so M1 trains in MHR space from the start. M0's remaining items are deferred: more seated-v2 recordings happen alongside, the face-cost check moves to M4. The rest of the order is still `[CLAUDE]`.

1. **M0: Baseline + evaluation harness.** Extract current tracking `[USER]`; implement metrics (§6); record a few seated test videos; measure baseline. Also re-check face blendshape cost with correct running mode.
2. **M1: Body keypoint lifter POC.** Plan: `dev-notes/plans/m1-body-lifter.md`. Decisions `[USER]` (2026-11-01): landmarks synthesized from fitted MHR poses with a session-calibrated noise and framing model (rendering stays in M6); small causal model over time-offset history, 6D output, rotation + FK position + joint-limit + velocity losses (over a GRU); CMU low-travel clips weighted to upper-body activity plus spliced seated motion, performer scales varied within MHR limits; new `train/` uv project (CUDA) and spike code moved into a shared package. MediaPipe pose keypoints → causal temporal head → rotations. Data: CMU + synthetic seated motion + simulated noise; short history window. Compare against baseline on real seated video. **Go/no-go point** for the learned approach.
3. **M2: Service skeleton** (can run parallel to M1). Python service owning camera; VMC over UDP + OSC over WebSocket; browser viewer with three-vrm; video-file source for repeatable tests; timestamps, sequence numbers, latest-wins.
4. **M3: MHR ↔ VRM spike** (can run parallel). Decide canonical skeleton.
5. **M4: Hand and face heads + merger.** Procedural hands; face head on top of MediaPipe blendshapes; modality dropout; residual merger; gaze angles; learned retargeting layer (§3.10a).
6. **M5: Root translation + calibration.**
7. **M6: Render-then-detect pipeline** (three.js first). Self-supervised reprojection on real video; adversarial motion prior.
8. **M7: Feature taps.** Convert MediaPipe models to PyTorch/ONNX; reimplement detect-track plumbing; feature-level heads (forearm twist).
9. **M8: Knobs** (loss-conditional training with FiLM).
10. **M9: Native core + bindings.**

Keep a **license register** (source + license of every asset, dataset, model, recording) from M0. `[CLAUDE]`
