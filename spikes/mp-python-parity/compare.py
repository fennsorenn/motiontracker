"""Compare landmark sets (Python vs WASM replay vs live recording) and m1 bones on each.

    uv run --project train python spikes/mp-python-parity/compare.py <ckpt> <dirA> <dirB> [<dirC> ...]
Each dir holds a landmarks.jsonl; frames are matched by t (ms). Pairs: A-B, A-C, ...: and B-C.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from mttrain.infer import OUTPUT_BONES, Runner, read_session

KEY = {"shoulders": [11, 12], "elbows": [13, 14], "wrists": [15, 16], "face": list(range(11))}


def load(d):
    rows = [json.loads(l) for l in (Path(d) / "landmarks.jsonl").read_text().splitlines() if l.strip()]
    return {int(round(r["t"] / 100.0)): r for r in rows}  # 100 ms grid (live times jitter)


def qang(a, b):
    d = np.abs(np.sum(a * b, -1)).clip(0, 1)
    return np.degrees(2 * np.arccos(d))


def main():
    ckpt, dirs = sys.argv[1], sys.argv[2:]
    L = {d: load(d) for d in dirs}
    r = Runner(ckpt)
    bones = {}
    for d in dirs:
        times, world, image, vis, present = read_session(d)
        rot, _ = r.predict(times, world, image, vis, present)
        b = r.vrm(rot)
        bones[d] = {int(round(t * 10)): {k: b[k][i] for k in OUTPUT_BONES if k in b} for i, t in enumerate(times)}
    for a, b in combinations(dirs, 2):
        ts = sorted(set(L[a]) & set(L[b]))
        print(f"\n== {Path(a).name} vs {Path(b).name}: {len(ts)} common frames")
        for s in ("pose", "face", "leftHand", "rightHand"):
            pa = np.array([s in L[a][t] for t in ts]); pb = np.array([s in L[b][t] for t in ts])
            print(f"  presence {s:9s} A {pa.mean():.3f} B {pb.mean():.3f} agree {np.mean(pa == pb):.3f}")
        both = [t for t in ts if "pose" in L[a][t] and "pose" in L[b][t] and "pose2d" in L[a][t] and "pose2d" in L[b][t]]
        W = lambda X, t: np.array(X[t]["pose"])
        I = lambda X, t: np.array(X[t]["pose2d"])
        wa, wb = np.stack([W(L[a], t) for t in both]), np.stack([W(L[b], t) for t in both])
        ia, ib = np.stack([I(L[a], t) for t in both]), np.stack([I(L[b], t) for t in both])
        for name, idx in KEY.items():
            e3 = np.linalg.norm(wa[:, idx, :3] - wb[:, idx, :3], axis=-1) * 100
            e2 = np.linalg.norm((ia[:, idx, :2] - ib[:, idx, :2]) * [320, 240], axis=-1)
            ev = np.abs(wa[:, idx, 3] - wb[:, idx, 3])
            print(f"  {name:9s} world cm med {np.median(e3):5.2f} p95 {np.percentile(e3, 95):5.2f} | image px med {np.median(e2):5.2f} p95 {np.percentile(e2, 95):5.2f} | vis med {np.median(ev):.3f}")
        bt = sorted(set(bones[a]) & set(bones[b]))
        print("  v10 bone angle diff (deg), median / p95 / max:")
        for k in ("chest", "neck", "head", "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand"):
            e = qang(np.stack([bones[a][t][k] for t in bt]), np.stack([bones[b][t][k] for t in bt]))
            print(f"    {k:14s} {np.median(e):5.2f} {np.percentile(e, 95):6.2f} {e.max():6.1f}")


main()
