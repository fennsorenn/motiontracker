"""The human side of live retargeting (`mtcore.relretarget.LiveRetarget`): MHR's mean body at rest, written into a
merger live bundle as `retarget.npz` so the runtime needs no pymomentum.

    uv run python -m mttrain.retarget_asset ../data/merger/runs/mg3/live

`retarget.npz`: `verts` (V, 3) float32 rest mesh vertices in metres (MHR world, same frame as the bundle's skeleton
rest state), `labels` (V,) the VRM bone each vertex counts for (its strongest joint's nearest mapped ancestor, as
`retarget_demo.human_body`), only vertices of anchor bones (`relretarget.ANCHORS`, plus eyes and jaw for the head).
Derived from MHR's assets (Apache 2.0).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.convert import MAP
from mtcore.mhr import load, skeleton_state
from mtcore.relretarget import ANCHORS, HEAD_PARTS


def vertex_bones(ch):
    jn = list(ch.skeleton.joint_names)
    parents = list(ch.skeleton.joint_parents)
    bone_of = {jn.index(MAP[b][0]): b for b in MAP if MAP[b][0] in jn}
    out = []
    for j in ch.skin_weights.to_dense(len(jn)).argmax(1):
        while j >= 0 and j not in bone_of:
            j = parents[j]
        out.append(bone_of.get(j, ""))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path, help="merger live bundle directory (with merger.json)")
    a = ap.parse_args()
    ch = load()
    sk = json.loads((a.bundle / "merger.json").read_text())["skeleton"]
    rest = skeleton_state(ch, np.zeros(len(ch.parameter_transform.names)))
    assert list(ch.skeleton.joint_names) == sk["joint_names"], "bundle skeleton is not this MHR"
    err = np.abs(np.asarray(sk["rest_state"]) - rest).max()
    assert err < 1e-3, f"bundle rest state differs from MHR's by {err}"
    labels = vertex_bones(ch)
    keep = np.isin(labels, list(ANCHORS) + list(HEAD_PARTS))
    verts = np.asarray(ch.mesh.vertices, dtype=np.float64)[keep] / 100.0
    np.savez_compressed(a.bundle / "retarget.npz", verts=verts.astype(np.float32), labels=labels[keep])
    print(f"{a.bundle / 'retarget.npz'}: {keep.sum()} of {len(labels)} vertices, "
          f"{(a.bundle / 'retarget.npz').stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
