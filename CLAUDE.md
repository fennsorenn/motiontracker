# CLAUDE.md

## Project
Learned live motion tracking for VRM 1.0 avatars: webcam → MediaPipe detectors → small learned temporal heads + merger → stable bone rotations, root translation, blendshapes and gaze → VMC/OSC output. Replaces an algorithmic MediaPipe pipeline whose jitter the owner couldn't fix.

**Read `PLAN.md` before doing anything.** It contains the full planning discussion, research findings, discarded approaches and open questions.

## Decision provenance: important
`PLAN.md` tags every decision:
- `[USER]`: confirmed by the project owner. Treat as requirements.
- `[USER-LEAN]`: owner preference, not final.
- `[CLAUDE]`: proposed by Claude in a previous planning session. **Not confirmed.** Do not treat as settled. When a task depends on one of these, say so, and ask the owner before committing to it in a way that's costly to undo.
- `[OPEN]`: unresolved.
- `[DISCARDED]`: rejected with a reason; don't re-propose without new information.
- `[CORRECTED]`: earlier claims later found wrong.

When the owner confirms or changes something, update the tag in `PLAN.md` (e.g. `[CLAUDE]` → `[USER]`) and note the date.

## Hard constraints (all `[USER]`)
- Latency is critical; real-time only. Strictly causal models.
- Lightweight: must run on busy hardware and ideally integrated graphics ("anything that runs MediaPipe").
- Permissive licensing for everything: data, models, assets. Owner wants a permissive release. Check `PLAN.md` §2.3 and §8 before adding any dataset or model.
- No owner-provided mocap. Only limited unlabeled video.
- Target VRM 1.0.
- Existing app is Node + three.js (three-vrm). Service is standalone first; language libs later.
- Person selection is out of scope (detector problem).

## Current state
Planning done; no code yet. Next step per suggested milestones (`PLAN.md` §10, unconfirmed order): M0, extract the owner's current tracking implementation as a baseline and build the evaluation harness.

## Housekeeping
- Maintain a license register (e.g. `docs/LICENSES-REGISTER.md`): source, license, link and usage for every asset, dataset, model and recording.
- Don't download or use any dataset/model marked non-commercial or unverified without asking.
