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
M0 in progress. Done: the baseline (`baseline/`, frozen copy of vspark's solver, bit-identical to vspark's engine), the session recorder (`tools/recorder/`) and the label-free evaluation harness (`eval/`). First real session recorded and measured (2026-11-01). M3 spike done (MHR canonical skeleton: go; `spikes/m3-mhr-vrm/`). Next: M1 body lifter in MHR space, with CMU → MHR fitting as data preparation; seated-v2 recordings alongside. See `dev-notes/ARCHITECTURE.md`.

## Housekeeping
- Maintain a license register (e.g. `docs/LICENSES-REGISTER.md`): source, license, link and usage for every asset, dataset, model and recording.
- Don't download or use any dataset/model marked non-commercial or unverified without asking.

# Project Documentation

Each project maintains a `dev-notes/` directory for developer-facing documentation:

- `dev-notes/ARCHITECTURE.md` — high-level overview of the project: major modules, how they fit together, key decisions and their rationale. Keep it lean; it should be enough to understand which modules are relevant, not a deep dive into each. Track the status of each module and major feature: implemented, WIP, or planned.
- `dev-notes/modules/<name>.md` — per-module detail: structure, patterns, how to extend it or add new types. Put specifics here, not in ARCHITECTURE.md.
- `dev-notes/modules/<name>/` — when a module contains multiple non-trivial implementations of the same concept (strategies, providers, handlers, adapters, etc.), use a subfolder instead. The module file becomes `<name>/index.md` and acts as an index: what the concept is, how implementations are registered or selected, and a list of the implementations with one-line descriptions. Each implementation gets its own file covering its behaviour, configuration, edge cases, and extension notes. File names should match the corresponding code identifiers. Don't pre-create files for planned implementations — list them in the index instead.

Cross-reference between module files when there are meaningful dependencies between modules. If one module's behaviour depends on or extends another, note it explicitly rather than leaving it implicit.

At the start of a new session, read `dev-notes/ARCHITECTURE.md` to orient yourself. When working in a specific module, read its module file if one exists.

Documentation is maintained by the `doc-updater` agent. Spawn it in the background (do not wait for it) in these situations, passing enough context for it to act without needing the session history:

- **Starting a task**: spawn with the task description and affected modules → it marks them WIP
- **Planning a feature**: spawn with the feature description and how it fits → it adds it as planned
- **Completing a task**: spawn with a summary of what changed, which modules were affected, and any new patterns or extension points → it updates statuses and module files

# Git

## Clean working tree before starting

Before starting any new feature branch — and before any non-trivial task — run `git status`. If there are uncommitted modifications or untracked files that look like in-progress work from a prior session, **stop and ask the user how to handle them** rather than working on top of them. Carrying someone else's uncommitted state into a new feature branch risks accidentally overwriting their work, mixing it into your commits, or having to revert files mid-task and losing it. Acceptable resolutions: (a) ask the user to commit/stash/discard, (b) stash them yourself with the user's explicit confirmation, (c) work on a different branch and come back later. Do not silently proceed.

## Branches

Work happens on feature branches, never directly on `dev` or `main`.

- Branch naming: `feature/<description>` or `bugfix/<description>`
- Create a branch at the start of any non-trivial task if one doesn't exist yet
- Feature branches merge into `dev` directly (no PR required)
- Merges from `dev` into `main` always go via a PR
- Every merge into `main` must be accompanied by a semver tag (e.g. `v1.2.0`), inferred from conventional commits since the last tag:
  - `fix:` → patch bump
  - `feat:` → minor bump
  - Any breaking change (`BREAKING CHANGE` in footer or `!` suffix) → major bump

## Commits

Use conventional commit messages: `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`, `test:`.

On a feature branch, commit proactively at natural stopping points — when a coherent unit of work is complete and the code is in a working state. Don't accumulate a session's worth of changes into one commit.

Before switching to a new task or context, run `git status`. If there are uncommitted changes, complete and commit them first rather than carrying dirty state across task boundaries.

## Committing only on AI-owned branches

Claude commits without asking for confirmation, but **only on AI-owned feature branches**. Never commit to `main`, `dev`, or a branch containing manual work.

A branch is **AI-owned** if it is a `feature/`, `bugfix/` or `claude/` branch and every commit on it since it left its base (`dev`, or `main` if there is no `dev`) carries a `Co-Authored-By:` trailer with `noreply@anthropic.com`. Check before the first commit of a session on that branch:

```
git log --format='%h%x09%(trailers:key=Co-Authored-By,valueonly,separator=%x2C)' dev..HEAD
```

Any commit without the Anthropic trailer means the branch has manual work. A branch with no commits since its base counts as AI-owned only if Claude created it in the current session.

If the current branch is `main`, `dev`, or not AI-owned, **branch off** first: create a new `feature/<description>` or `bugfix/<description>` from the current HEAD and commit there. Never add commits to the user's branch. Mention the new branch name to the user when you create it.

Never skip hooks (`--no-verify`).

# Handoffs

Implementation work can be handed off from a planning session to another session: a cloud session at claude.ai/code (`cloud-handoff` skill) or a separate local session (`start-claude-session` skill). Handoff plans go in `dev-notes/plans/<descriptive-name>.md`, based on `dev-notes/plans/plan-template.md`.
