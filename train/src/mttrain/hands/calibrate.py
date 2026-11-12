"""MediaPipe hand landmarks vs truth on rendered hands: what the synthetic observations must reproduce.

    uv run python -m mttrain.hands.calibrate data/m4/renders/<set> [--json out.json]

Per detected hand frame, MediaPipe's world landmarks and the true MHR hand points go to the canonical
palm-frame form (`mtcore.hand_features.canonical`), so only shape errors remain. Reports: per-point
bias and noise (palm-frame axes, in units of the wrist -> middle knuckle length), mirrored shapes
(palm and back confused), frame-to-frame correlation of the errors (10 Hz), error against how the palm
faces the camera, and the detection rate against the hand's size in the image.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.hand_features import canonical, palm_frame

from .evaluate import Truth
from .skeleton import HandFK

MIRROR = np.array([1.0, 1.0, -1.0])  # in the canonical frame: palm and back swapped


def collect(set_dir, fk=None, landmarks="landmarks.jsonl"):
    """Per hand-frame records: seq, side, t, detected, true canonical (21, 3), MediaPipe canonical or None,
    hand size in the image (fraction of the frame width), palm-to-camera angle (deg)."""
    fk = fk or HandFK()
    rows = []
    for d in sorted(Path(set_dir).glob("seq*")):
        z = np.load(d / "seq.npz")
        Rc, tc, fx = z["R"], z["t"], float(z["fx"])
        truth = Truth(d, fk)
        for line in open(d / landmarks):
            f = json.loads(line)
            for side, S in (("l", "left"), ("r", "right")):
                P = truth.points(side, f["t"])
                C = (truth.posed_points(side, f["t"]) - tc) @ Rc.T  # camera frame: x right, y down, z forward
                size = fx * np.linalg.norm(C[9, :2] - C[0, :2]) / max(C[0, 2], 1e-3)
                _, R, _ = palm_frame(C * ([-1, 1, 1] if side == "r" else 1))
                facing = np.degrees(np.arccos(np.clip(abs(R[2, 2]), 0, 1)))  # 0: palm or back faces the camera
                mp = f.get(f"{S}Hand")
                rows.append({"seq": d.name, "side": side, "t": f["t"], "true": canonical(P, side),
                             "mp": canonical(np.array(mp), side) if mp else None, "size": size, "facing": facing,
                             "in_view": bool((C[:, 2] > 0.05).all() and (np.abs(fx * C[:, 0] / C[:, 2]) < 0.5).all()
                                             and (np.abs(fx * 4 / 3 * C[:, 1] / C[:, 2]) < 0.5).all())})
    return rows


def analyse(rows):
    det = [r for r in rows if r["mp"] is not None]
    T = np.stack([r["true"] for r in det])
    M = np.stack([r["mp"] for r in det])
    e_plain = np.linalg.norm(M - T, axis=-1).mean(-1)
    e_mirror = np.linalg.norm(M * MIRROR - T, axis=-1).mean(-1)
    flipped = e_mirror < 0.6 * e_plain
    ok = ~flipped
    E = M[ok] - T[ok]
    # linear fit per coordinate, MediaPipe = a + b * truth (b < 1: pulled toward an average hand), and the
    # residuals' covariance across all 63 coordinates: the synthetic observation model (mttrain.hands.observe)
    Tf, Mf = T[ok].reshape(len(E), -1), M[ok].reshape(len(E), -1)
    tm, mm = Tf.mean(0), Mf.mean(0)
    var = ((Tf - tm) ** 2).mean(0)
    b = np.where(var > 1e-8, ((Tf - tm) * (Mf - mm)).mean(0) / np.maximum(var, 1e-8), 1.0)
    a = mm - b * tm
    resid = Mf - (a + b * Tf)
    fit = {"a": a.tolist(), "b": b.tolist(), "cov": np.cov(resid.T).tolist(), "n": int(len(E))}
    out = {
        "fit": fit,
        "frames": len(rows), "in_view": int(sum(r["in_view"] for r in rows)),
        "detected": len(det), "detected_in_view": int(sum(r["in_view"] for r in det)),
        "flipped": float(flipped.mean()),
        "bias": E.mean(0).tolist(), "std": E.std(0).tolist(),
        "err_median": float(np.median(np.linalg.norm(E, axis=-1))),
    }
    # frame-to-frame correlation of the errors within a run of consecutive detections
    by = {}
    for r, f in zip(det, flipped):
        if not f:
            by.setdefault((r["seq"], r["side"]), []).append((r["t"], r["mp"] - r["true"]))
    a, b = [], []
    for v in by.values():
        for (t0, e0), (t1, e1) in zip(v, v[1:]):
            if t1 - t0 < 150:
                a.append((e0 - E.mean(0)).ravel()); b.append((e1 - E.mean(0)).ravel())
    a, b = np.array(a), np.array(b)
    out["lag1_corr"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    out["fit"]["lag1"] = out["lag1_corr"]
    out["fit"]["flipped"] = out["flipped"]
    out["fit"]["resid_median"] = float(np.median(np.linalg.norm(resid.reshape(-1, 21, 3), axis=-1)))
    # error and flips against how the palm faces the camera
    facing = np.array([r["facing"] for r in det])
    out["by_facing"] = [{"deg": f"{lo}-{hi}", "n": int(m.sum()), "flipped": float(flipped[m].mean()),
                         "err": float(np.median(e_plain[m & ok]))}
                        for lo, hi in ((0, 30), (30, 60), (60, 90)) for m in [(facing >= lo) & (facing < hi)] if m.sum()]
    # detection against hand size in the image (in view only)
    vis = [r for r in rows if r["in_view"]]
    size = np.array([r["size"] for r in vis])
    got = np.array([r["mp"] is not None for r in vis])
    edges = np.quantile(size, [0, 0.25, 0.5, 0.75, 1.0])
    out["by_size"] = [{"size": f"{lo:.3f}-{hi:.3f}", "detected": float(got[(size >= lo) & (size <= hi)].mean())}
                      for lo, hi in zip(edges, edges[1:])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--landmarks", default="landmarks.jsonl", help="landmark file name in each sequence")
    a = ap.parse_args()
    out = analyse(collect(a.set, landmarks=a.landmarks))
    names = ["wrist", "thumb CMC", "thumb MCP", "thumb IP", "thumb tip"] + [
        f"{f} {j}" for f in ("index", "middle", "ring", "pinky") for j in ("MCP", "PIP", "DIP", "tip")]
    print(f"{out['frames']} hand-frames, {out['in_view']} in view; detected {out['detected']} "
          f"({out['detected_in_view']} in view); mirrored {100 * out['flipped']:.0f}%; "
          f"median point error {out['err_median']:.3f} palm lengths; lag-1 error correlation {out['lag1_corr']:.2f}")
    print("per point (palm lengths): bias fwd/thumbward/palm, std fwd/thumbward/palm")
    for n, b, s in zip(names, out["bias"], out["std"]):
        print(f"  {n:11s} {b[0]:+.3f} {b[1]:+.3f} {b[2]:+.3f}   {s[0]:.3f} {s[1]:.3f} {s[2]:.3f}")
    f = out["fit"]
    print(f"linear fit: slope b median {np.median(f['b']):.2f} (fwd {np.median(f['b'][0::3]):.2f}, thumbward "
          f"{np.median(f['b'][1::3]):.2f}, palm {np.median(f['b'][2::3]):.2f}); residual median {f['resid_median']:.3f} palm lengths")
    print("by palm facing:", out["by_facing"])
    print("detection by hand size (wrist-middle knuckle / frame width):", out["by_size"])
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
