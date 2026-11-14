"""Summary table of a merger evaluation on the rendered sets (tools/merger/eval.sh): per set the median global
rotation error of head, lower arms and hands (and head onset lag) for the body model, base pose and merger.

    uv run python -m mttrain.merger.evalsummary data/merger/eval/<run>
"""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
BONES = ["head", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand"]
SYS = ["m1", "base", "merger"]
lines = ["| set | bone | " + " | ".join(SYS) + " |", "|---|---|" + "---|" * len(SYS)]
summary = {}
for setdir in sorted(p for p in root.iterdir() if p.is_dir()):
    reps = [json.load(open(f)) for f in sorted(setdir.glob("seq*/report/report.json"))]
    if not reps:
        continue
    for b in BONES + ["hands", "lag_head"]:
        row = []
        for s in SYS:
            vals = []
            for r in reps:
                g = r["systems"][s].get("ground_truth", {})
                if b == "hands":
                    v = np.mean([g["global_rotation_error"][f"global:{h}"]["median"] for h in ("leftHand", "rightHand")])
                elif b == "lag_head":
                    v = g.get("onset_lag", {}).get("global:head", {}).get("lag_ms", np.nan)
                else:
                    v = g["global_rotation_error"][f"global:{b}"]["median"]
                vals.append(v)
            row.append(float(np.nanmean(vals)))
        summary.setdefault(setdir.name, {})[b] = dict(zip(SYS, row))
        if b in ("head", "leftLowerArm", "rightLowerArm", "hands", "lag_head"):
            lines.append(f"| {setdir.name} ({len(reps)}) | {b} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
(root / "summary.json").write_text(json.dumps(summary, indent=1))
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
