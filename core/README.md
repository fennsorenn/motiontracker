# mtcore

Shared Python code, moved out of the M3 spike (`spikes/m3-mhr-vrm/`) for M1 and later stages.
Consumers add it as an editable path dependency (`mtcore = { path = "../core", editable = true }`)
and choose their own torch build (CPU here and in the spike, CUDA in `train/`).

| Module | What |
|---|---|
| `paths.py` | Repo and data locations; `MT_DATA` overrides the gitignored `data/`. |
| `mhr.py` | MHR rig via pymomentum (no blendshapes: `MHR.from_files` segfaults), FK, `[Limits]` parser, `FLEX_T`. |
| `convert.py` | MHR skeleton state → VRM 1.0 normalized local rotations (`MhrToVrm`), VRM FK, quaternion helpers. |
| `vrm.py` | Minimal VRM 0.x/1.0 humanoid reader (0.x turned to face +Z). |
| `cmu.py` | CMU ASF/AMC reader + FK (cm). |
| `cmu_fit.py` | CMU clip → MHR: shared performer scales + per-frame pose (pymomentum sequence IK). |

```sh
uv sync
uv run pytest     # tests skip when data/mhr, data/cmu or ../volleyball avatars are missing
```
