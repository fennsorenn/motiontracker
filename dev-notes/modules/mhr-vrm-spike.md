# mhr-vrm-spike (implemented, M3 done)

Location: `spikes/m3-mhr-vrm/` (uv project; CPU PyTorch, `pymomentum`, `mhr`). Usage: `spikes/m3-mhr-vrm/README.md`. Results and numbers: `dev-notes/plans/m3-mhr-vrm-spike.md` and `PLAN.md` §3.10 (not copied here). Outcome: go for MHR as canonical skeleton.

Spike code, not production. Pieces likely reused in M1 (CMU → MHR data prep) and later in the MHR → VRM output stage.

## Files
| File | Role |
|---|---|
| `convert.py` | MHR skeleton state → VRM 1.0 normalized local rotations. |
| `vrm_skeleton.py` | VRM 0.x / 1.0 skeleton reader. VRM 0.x models are turned to face +Z (the VRM 1.0 convention). |
| `cmu.py` | CMU ASF/AMC reader + forward kinematics. |
| `fit_cmu.py` | Fits CMU clips to MHR with pymomentum `solve_sequence_ik`. |
| `scan_cmu.py` | Scans the CMU set (output: `cmu_scan.json`). |
| `eval_cmu_roundtrip.py` | Round-trip / spatial-relation evaluation on CMU clips. |

Other scripts in the folder (`fit_avatars.py`, `dump_skeleton.py`, `test_*.py`) are not described here yet.

## Conversion (MHR → VRM 1.0)
- Per bone: `G_vrm = Q_mhr * Q_mhr_rest^-1 * D` (MHR rotation relative to MHR rest, times a per-bone term `D`; see `convert.py` for how `D` is built). Output is VRM 1.0 normalized local rotations.
- Forearm twist: MHR's forearm twist is moved onto VRM `lowerArm` by a swing-twist split.
- MHR flexible joint translations are kept at zero (rotation-only output).

## CMU → MHR fitting
- `solve_sequence_ik`: performer scales shared across the sequence, pose solved per frame.
- The end of CMU's `wrist` bone is the real wrist (mapping detail for the wrist target).
- Hand and finger scales are frozen.

## Known issues
- `MHR.from_files` segfaults inside pymomentum's `with_blend_shape` (pymomentum 0.1.114, CPU). The spike loads the rig without blendshapes. Needs a fix or workaround before identity/shape fitting relies on blendshapes.

## Data
Gitignored `data/`: `data/cmu` (CMU mocap), `data/mhr` (MHR assets). Licensing: `docs/LICENSES-REGISTER.md`.
