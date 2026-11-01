# M3 spike: MHR ↔ VRM 1.0

Plan and findings: `dev-notes/plans/m3-mhr-vrm-spike.md`. Needs `data/mhr/assets/` (MHR release
assets), `data/cmu/all_asfamc/` (CMU ASF/AMC) and the VRoid samples in `../volleyball`.

```sh
uv sync                                   # CPU PyTorch + pymomentum + mhr
uv run python dump_skeleton.py            # MHR rest skeleton, scale parameter effects
uv run python fit_avatars.py              # VRoid proportions vs MHR scale parameters/limits
uv run python test_roundtrip.py AVATAR    # bone directions after MHR -> VRM (random poses)
uv run python test_positions.py 0.5       # end-effector error from joints VRM lacks
uv run python fit_cmu.py 2 1              # fit MHR to one CMU clip
uv run python scan_cmu.py                 # rank CMU clips by upper-body activity
uv run python eval_cmu_roundtrip.py       # CMU -> MHR -> avatars: spatial relations
```

| File | What |
|---|---|
| `convert.py` | MHR skeleton state → VRM 1.0 normalized local rotations (`MhrToVrm`), VRM FK. |
| `vrm_skeleton.py` | Minimal VRM 0.x/1.0 humanoid reader (0.x turned to face +Z). |
| `cmu.py` | CMU ASF/AMC reader + FK. |
| `fit_cmu.py` | pymomentum sequence IK: shared performer scales + per-frame pose. |

Known issue: `MHR.from_files` segfaults (pymomentum-cpu 0.1.114, torch 2.8 CPU) in
`Character.with_blend_shape`; the spike loads the rig without blendshapes instead.
