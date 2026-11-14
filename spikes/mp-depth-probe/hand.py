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


def affine_scale(p_img, P_world):
    """Weak-perspective scale with the hand's orientation fitted: (p - p_mean) = M (P - P_mean),
    M 2x3 least squares; for a scaled orthographic projection both singular values of M equal the
    scale, so their mean is orientation-free. Also returns their ratio (1 = consistent fit)."""
    A, B = P_world - P_world.mean(0), p_img - p_img.mean(0)
    M = np.linalg.lstsq(A, B, rcond=None)[0].T
    sv = np.linalg.svd(M, compute_uv=False)
    return sv.mean(), sv[1] / sv[0], sv[0]


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
            sa, ratio, smax = affine_scale(p, world)
            # how edge-on the palm is: |normal . view axis| from the world landmarks' palm plane, rotated
            # into the camera with the fitted M (3rd axis = cross of its rows)
            A, B = world - world.mean(0), p - p.mean(0)
            M = np.linalg.lstsq(A, B, rcond=None)[0].T
            r1, r2 = M[0] / np.linalg.norm(M[0]), M[1] / np.linalg.norm(M[1])
            view = np.cross(r1, r2); view /= np.linalg.norm(view)
            pa = world[PALM] - world[PALM].mean(0)
            normal = np.linalg.svd(pa)[2][-1]
            facing = abs(normal @ view)  # 1 = palm faces the camera, 0 = edge-on
            rows.append((d.name, meta["gaps"][g], pres, z_true, fx_px / scale(p, world, list(range(21))),
                         fx_px / scale(p, world, PALM), err2d, fx_px / sa, facing, fx_px / smax))
    cfg = np.array([r[0] for r in rows])
    gap, pres, zt, z3, zp, e2, za, facing, zx = (np.array([r[i] for r in rows], dtype=float) for i in range(1, 10))
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
    for name, z in (("3D spread", z3), ("palm", zp), ("fitted orientation", za), ("fitted, largest extent", zx)):
        t, e = within(zt)[keep] * 100, within(z)[keep] * 100
        slope = (t * e).sum() / (t * t).sum()
        r = np.corrcoef(t, e)[0, 1]
        print(f"  {name}: within-configuration slope {slope:+.2f}, correlation {r:+.2f}, R2 of a fitted line {r * r:.2f}; "
              f"within error median {np.median(np.abs(e - t)):.1f} cm")
    m = ok
    print(f"\nabsolute hand distance, all frames with the hand found (n={m.sum()}): median |error| 3D {np.median(np.abs(z3 - zt)[m]) * 100:.1f} cm, "
          f"palm {np.median(np.abs(zp - zt)[m]) * 100:.1f} cm; median signed 3D {np.median((z3 - zt)[m]) * 100:+.1f} cm; "
          f"relative |error| {np.median(np.abs(z3 / zt - 1)[m]):.1%}")
    print(f"orientation-fitted: median |error| {np.median(np.abs(za - zt)[m]) * 100:.1f} cm, signed {np.median((za - zt)[m]) * 100:+.1f} cm")
    print(f"fitted, largest extent: median |error| {np.median(np.abs(zx - zt)[m]) * 100:.1f} cm, signed {np.median((zx - zt)[m]) * 100:+.1f} cm")
    print("median |error| cm by palm facing (|cos| palm normal vs view axis), 3D spread / orientation-fitted / largest extent:")
    for lo, hi in ((0, 0.4), (0.4, 0.7), (0.7, 1.01)):
        b = m & (facing >= lo) & (facing < hi)
        if b.any():
            print(f"  {lo:.1f}-{hi:.1f} (n={b.sum():3d}): {np.median(np.abs(z3 - zt)[b]) * 100:5.1f} / {np.median(np.abs(za - zt)[b]) * 100:5.1f}"
                  f" / {np.median(np.abs(zx - zt)[b]) * 100:5.1f}   signed {np.median((z3 - zt)[b]) * 100:+5.1f} / "
                  f"{np.median((za - zt)[b]) * 100:+5.1f} / {np.median((zx - zt)[b]) * 100:+5.1f}")
    print("hand found, by gap: " + ", ".join(f"{g:.1f} cm {ok[gap == g].mean():.0%}" for g in np.unique(gap)))
    print("true within-configuration depth range (max - min), median: "
          f"{np.median([np.ptp(zt[cfg == c]) for c in np.unique(cfg)]) * 100:.1f} cm")
    # log-ratio form: relative depth change ~ -log(size) change; scale-free (no metric hand needed)
    print("\n(For reference: pose-model probes reached within-configuration R2 <= 0.22; MediaPipe pose output slope -0.26.)")


if __name__ == "__main__":
    main()
