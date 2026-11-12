"""MediaPipe hand failures on rendered sessions: identity swaps, doubled hands, palm/back flips.

    uv run python -m mttrain.hands.failures data/m4/renders/<set> [<set> ...] [--json out.json]

Per detected hand: which true hand it sits on (its image wrist and middle knuckle against both true hands
projected with the render camera; MediaPipe sees the unmirrored render, see mttrain.render), and whether
its shape is mirrored (palm and back confused; canonical shape closer to the mirrored truth). Reports
rates, run lengths at 10 Hz, and when they happen (hands close together, palm facing).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.hand_features import canonical

from .calibrate import MIRROR
from .evaluate import Truth
from .skeleton import HandFK


def project(C, fx, aspect=4 / 3):
    return np.stack([0.5 + fx * C[..., 0] / C[..., 2], 0.5 + fx * aspect * C[..., 1] / C[..., 2]], -1)


def frames(set_dirs, fk=None):
    fk = fk or HandFK()
    rows = []
    for set_dir in set_dirs:
        for d in sorted(Path(set_dir).glob("seq*")):
            z = np.load(d / "seq.npz")
            Rc, tc, fx = z["R"], z["t"], float(z["fx"])
            truth = Truth(d, fk)
            for line in open(d / "landmarks.jsonl"):
                f = json.loads(line)
                img = {s: project((truth.posed_points(s, f["t"]) - tc) @ Rc.T, fx) for s in "lr"}
                gap = float(np.linalg.norm(img["l"][[0, 9]].mean(0) - img["r"][[0, 9]].mean(0)))
                size = {s: float(np.linalg.norm(img[s][9] - img[s][0])) for s in "lr"}
                for side, S in (("l", "left"), ("r", "right")):
                    h2 = f.get(f"{S}Hand2d")
                    if not h2:
                        rows.append({"seq": f"{Path(set_dir).name}/{d.name}", "side": side, "t": f["t"], "det": False,
                                     "gap": gap / max(size[side], 1e-6)})
                        continue
                    p = np.array(h2)[[0, 9], :2].mean(0)
                    dist = {s: float(np.linalg.norm(p - img[s][[0, 9]].mean(0))) / max(size[s], 1e-6) for s in "lr"}
                    owner = min(dist, key=dist.get)
                    true_c = canonical(truth.points(owner, f["t"]), owner)
                    mp_c = canonical(np.array(f[f"{S}Hand"]), side) if f.get(f"{S}Hand") else None
                    flipped = None
                    if mp_c is not None:
                        e, em = (np.linalg.norm(x - true_c, axis=-1).mean() for x in (mp_c, mp_c * MIRROR))
                        flipped = bool(em < 0.6 * e)
                    rows.append({"seq": f"{Path(set_dir).name}/{d.name}", "side": side, "t": f["t"], "det": True,
                                 "owner": owner, "swapped": owner != side, "dist": dist[owner], "flipped": flipped,
                                 "gap": gap / max(size[side], 1e-6)})
    return rows


def runs(flags):
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def analyse(rows):
    det = [r for r in rows if r["det"]]
    out = {"hand_frames": len(rows), "detected": len(det)}
    out["swapped"] = float(np.mean([r["swapped"] for r in det]))
    # doubled: both detections of a frame on the same true hand
    by_t = {}
    for r in det:
        by_t.setdefault((r["seq"], r["t"]), []).append(r)
    both = [v for v in by_t.values() if len(v) == 2]
    out["frames_with_two_detections"] = len(both)
    out["doubled"] = float(np.mean([v[0]["owner"] == v[1]["owner"] for v in both])) if both else 0.0
    fl = [r for r in det if r["flipped"] is not None and not r["swapped"]]
    out["flipped"] = float(np.mean([r["flipped"] for r in fl]))
    seqs = {}
    for r in rows:
        seqs.setdefault((r["seq"], r["side"]), []).append(r)
    swap_runs, flip_runs = [], []
    for v in seqs.values():
        swap_runs += runs([r["det"] and r["swapped"] for r in v])
        flip_runs += runs([bool(r["det"] and r.get("flipped")) for r in v])
    out["swap_runs"] = {"n": len(swap_runs), "median": float(np.median(swap_runs)) if swap_runs else 0,
                        "max": int(max(swap_runs)) if swap_runs else 0}
    out["flip_runs"] = {"n": len(flip_runs), "median": float(np.median(flip_runs)) if flip_runs else 0,
                        "p90": float(np.percentile(flip_runs, 90)) if flip_runs else 0}
    # by how close the hands are (distance between hand centres in hand lengths)
    gaps = np.array([r["gap"] for r in det])
    out["by_gap"] = []
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 1e9)):
        m = (gaps >= lo) & (gaps < hi)
        if m.sum():
            sub = [r for r, k in zip(det, m) if k]
            out["by_gap"].append({"gap": f"{lo}-{hi:g}", "n": int(m.sum()), "swapped": float(np.mean([r["swapped"] for r in sub])),
                                  "flipped": float(np.mean([bool(r["flipped"]) for r in sub]))})
    lost = [r for r in rows if not r["det"]]
    out["lost_by_gap"] = [{"gap": g["gap"], "lost": float(np.mean([(lo <= r["gap"] < hi) for r in lost]))}
                          for g, (lo, hi) in zip(out["by_gap"], ((0, 1.5), (1.5, 3), (3, 6), (6, 1e9)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = frames(a.sets)
    out = analyse(rows)
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
