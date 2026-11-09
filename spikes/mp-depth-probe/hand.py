"""Follow-up: does MediaPipe's hand landmark model (Holistic bundle, own 224 px hand crop) carry the
hand's depth through its apparent size? Same 320 contrast frames as probe.py.

    uv run python hand.py

Crop: from the true pose landmarks of the active hand (wrist, index/pinky knuckles), then refined
from the hand model's own landmarks for 4 rounds (box around the landmarks, x2, rotated wrist ->
middle knuckle up, shifted 0.1 toward the fingers), as hand tracking would.
Distance from size: the model's world landmarks are metric, its image landmarks are pixels, so the
weak-perspective scale s (px per m) gives z ~ fx_px / s. Two variants: 3D spread (orientation-free
but foreshortened) and the palm (wrist, index/middle/pinky knuckles) only.
Evaluated within configurations (hand moves along the face normal, everything else fixed).
"""

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

from probe import D, LIX, LPI, LWR, RIX, RPI, RWR, crop

MODEL = D / "model" / "x" / "hand_landmarks_detector.tflite"
PALM = [0, 5, 9, 13, 17]


def run(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x[None].astype(np.float32))
    it.invoke()
    o = {d["name"]: it.get_tensor(d["index"])[0] for d in it.get_output_details()}
    return o["Identity"].reshape(21, 3), float(o["Identity_1"][0]), o["Identity_3"].reshape(21, 3)


def roi_from_hand(p):
    up = p[9] - p[0]
    th = np.arctan2(up[0], -up[1])
    cs, sn = np.cos(th), np.sin(th)
    rot = np.array([[cs, sn], [-sn, cs]])  # image -> upright crop axes
    q = (p - p[0]) @ rot.T
    lo, hi = q.min(0), q.max(0)
    size = 2.0 * (hi - lo).max()
    c_up = 0.5 * (lo + hi) + [0, -0.1 * size]
    c = p[0] + c_up @ rot
    return c, size, th


def scale(p_img, P_world, idx):
    a, b = p_img[idx], P_world[idx]
    return np.sqrt(((a - a.mean(0)) ** 2).sum() / ((b - b.mean(0)) ** 2).sum())


def main():
    it = Interpreter(model_path=str(MODEL))
    it.allocate_tensors()
    rows = []
    for d in sorted((D / "configs").glob("c*")):
        meta = np.load(d / "meta.npz")
        side = str(meta["side"])
        wr, pi, ix = (LWR, LPI, LIX) if side == "l" else (RWR, RPI, RIX)
        fx_px = float(meta["fx"]) * 640
        for g in range(len(meta["gaps"])):
            img = np.asarray(Image.open(d / "frames" / f"{g:05d}.png").convert("RGB"), dtype=np.float64) / 255.0
            p2 = meta["pose2d"][g, :, :2] * [640, 480]
            k = 0.5 * (p2[pi] + p2[ix])
            up = k - p2[wr]
            roi = (p2[wr] + 0.6 * up, 4.0 * np.linalg.norm(up), np.arctan2(up[0], -up[1]))
            for _ in range(4):
                x, to_img, _ = crop(img, *roi, n=224)
                lm, pres, world = run(it, x)
                p = to_img(lm[:, :2])
                roi = roi_from_hand(p)
            x, to_img, _ = crop(img, *roi, n=224)
            lm, pres, world = run(it, x)
            p = to_img(lm[:, :2])
            z_true = 0.5 * (meta["cam_xyz"][g, pi] + meta["cam_xyz"][g, ix])[2]
            err2d = np.linalg.norm(p[0] - p2[wr])
            rows.append((d.name, meta["gaps"][g], pres, z_true, fx_px / scale(p, world, list(range(21))),
                         fx_px / scale(p, world, PALM), err2d))
    cfg = np.array([r[0] for r in rows])
    gap, pres, zt, z3, zp, e2 = (np.array([r[i] for r in rows], dtype=float) for i in range(1, 7))
    print(f"{len(rows)} frames; presence > 0.5: {(pres > 0.5).mean():.0%}; wrist 2D vs true pose wrist: median {np.median(e2):.0f} px")
    ok = pres > 0.5
    keep = np.array([ok[cfg == c].all() for c in cfg])  # configurations with the hand found at every gap
    print(f"configurations with the hand found at all 4 gaps: {len(np.unique(cfg[keep]))} / {len(np.unique(cfg))}")

    def within(v):
        out = v.copy()
        for c in np.unique(cfg):
            m = cfg == c
            out[m] = v[m] - v[m].mean()
        return out
    print(f"\n{'gap cm':>7s} {'true hand z':>12s} {'est (3D)':>9s} {'est (palm)':>11s}   (cm, medians over kept frames)")
    for g in np.unique(gap):
        m = keep & (gap == g)
        print(f"{g:7.1f} {np.median(zt[m]) * 100:12.1f} {np.median(z3[m]) * 100:9.1f} {np.median(zp[m]) * 100:11.1f}")
    for name, z in (("3D spread", z3), ("palm", zp)):
        t, e = within(zt)[keep] * 100, within(z)[keep] * 100
        slope = (t * e).sum() / (t * t).sum()
        r = np.corrcoef(t, e)[0, 1]
        print(f"  {name}: within-configuration slope {slope:+.2f}, correlation {r:+.2f}, R2 of a fitted line {r * r:.2f}; "
              f"within error median {np.median(np.abs(e - t)):.1f} cm")
    m = ok
    print(f"\nabsolute hand distance, all frames with the hand found (n={m.sum()}): median |error| 3D {np.median(np.abs(z3 - zt)[m]) * 100:.1f} cm, "
          f"palm {np.median(np.abs(zp - zt)[m]) * 100:.1f} cm; median signed 3D {np.median((z3 - zt)[m]) * 100:+.1f} cm; "
          f"relative |error| {np.median(np.abs(z3 / zt - 1)[m]):.1%}")
    print("hand found, by gap: " + ", ".join(f"{g:.1f} cm {ok[gap == g].mean():.0%}" for g in np.unique(gap)))
    print("true within-configuration depth range (max - min), median: "
          f"{np.median([np.ptp(zt[cfg == c]) for c in np.unique(cfg)]) * 100:.1f} cm")
    # log-ratio form: relative depth change ~ -log(size) change; scale-free (no metric hand needed)
    print("\n(For reference: pose-model probes reached within-configuration R2 <= 0.22; MediaPipe pose output slope -0.26.)")


if __name__ == "__main__":
    main()
