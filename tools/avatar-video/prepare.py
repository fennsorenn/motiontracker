"""Prepare the pose tracks for a side-by-side avatar video of a recorded session.

    uv run --project eval python tools/avatar-video/prepare.py <session dir> <out dir> \
        --system baseline-rendered --system m1-v8-glide=data/m1/runs/m1-v8/session.poses.jsonl

Systems use the harness syntax (`mteval evaluate --system`), so each panel shows exactly the track
the harness measures. Writes <out dir>/tracks.json: session segments plus one panel per system
(times ms, VRM normalized bone rotations, null where absent). By default limb bones only when every
system outputs them, so no panel gets parts (fingers, ...) the others lack; the torso/neck/head
chain is kept per system, because systems split the same head pose differently across it (the
baseline has no hips, upperChest or head bone; see `mteval.metrics.with_head_global`).
`--all-bones` keeps each system's full set.

`--blendshapes NAME=FILE` adds ARKit blendshapes to panel NAME: a JSONL file with `t` (ms) and
`blendshapes` ({ARKit name: 0..1}) per line, such as the runtime's `--dump-landmarks` output. The
renderer drives the avatar's face with them (render.mjs `--arkit-map`).

`--hand-insets`: every avatar panel gets close-ups of both hands along the bottom (for finger tracks).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mteval.metrics import HEAD_CHAIN
from mteval.session import load_session
from mteval.systems import run_system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--system", action="append", required=True)
    ap.add_argument("--all-bones", action="store_true")
    ap.add_argument("--blendshapes", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--hand-insets", action="store_true", help="close-ups of both hands in every avatar panel")
    a = ap.parse_args()
    session = load_session(a.session)
    results = [run_system(s, session) for s in a.system]
    common = set.intersection(*(set(r.track.bones) for r in results))
    panels = []
    for r in results:
        names = sorted(r.track.bones if a.all_bones else [n for n in r.track.bones if n in common or n in HEAD_CHAIN])
        bones = {n: [None if np.isnan(q).any() else [round(float(v), 6) for v in q] for q in r.track.bones[n]] for n in names}
        panels.append({"name": r.name, "notes": r.notes, "t": [float(t) for t in r.track.times], "bones": bones})
        print(f"{r.name}: {len(r.track.times)} frames, {len(names)} bones ({', '.join(r.notes)})")
    for spec in a.blendshapes:
        name, path = spec.split("=", 1)
        panel = next((p for p in panels if p["name"] == name), None)
        if panel is None:
            raise SystemExit(f"--blendshapes: no panel named {name!r}")
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        panel["arkit"] = {"t": [float(r["t"]) for r in rows], "values": [r.get("blendshapes") for r in rows]}
        print(f"{name}: blendshapes on {sum(v is not None for v in panel['arkit']['values'])}/{len(rows)} frames")
    meta = json.loads((Path(a.session) / "session.json").read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps({"segments": meta.get("segments", []), "panels": panels,
                                                 "handInsets": a.hand_insets}))
    dropped = sorted(set.union(*(set(r.track.bones) for r in results)) - common - set(HEAD_CHAIN))
    if dropped and not a.all_bones:
        print(f"bones not shared by every system, left at rest: {', '.join(dropped)}")


if __name__ == "__main__":
    main()
