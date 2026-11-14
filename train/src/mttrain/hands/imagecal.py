"""MediaPipe's image-space hand landmarks vs truth on rendered sessions: what the pose half of the pair model
(hand orientation and position in the camera frame) has to cope with.

    uv run python -m mttrain.hands.imagecal data/m4/renders/<set> [--landmarks landmarks-hl.jsonl] [--json out.json]

Image landmarks (x, y normalized to the frame, z relative depth on the x scale) -> 3D points on one scale
(`image_points`); the hand's orientation in the camera frame is the palm frame of those points
(`mtcore.hand_features.palm_frame`; right hands mirrored into left ones). Truth: the posed MHR hand projected
with the render camera, the same way. Reports orientation error (deg), how often it is mirrored, point error
in hand lengths (x/y and z), position error, all for detections on the right hand (identity swaps excluded).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.hand_features import palm_frame

from .evaluate import Truth
from .failures import project
from .skeleton import HandFK

ASPECT = 4 / 3  # frame width / height


def image_points(L, side):
    """MediaPipe image hand landmarks (..., 21, 3) -> points on one scale (x right, y down, z away; units of
    the frame width), right hands mirrored (x -> -x)."""
    L = np.asarray(L, dtype=np.float64)
    P = np.stack([L[..., 0] - 0.5, (L[..., 1] - 0.5) / ASPECT, L[..., 2]], -1)
    return P * ([-1.0, 1.0, 1.0] if side == "r" else 1.0)


def truth_image(C, fx, side):
    """Camera-frame points (21, 3) -> the same form as image_points: projected x/y, z relative to the wrist."""
    z0 = C[0, 2]
    x = fx * C[:, 0] / C[:, 2]
    y = fx * C[:, 1] / C[:, 2]
    z = fx * (C[:, 2] - z0) / z0
    return np.stack([x, y, z], -1) * ([-1.0, 1.0, 1.0] if side == "r" else 1.0)


def rot_angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def collect(set_dir, landmarks="landmarks.jsonl", fk=None):
    fk = fk or HandFK()
    rows = []
    for d in sorted(Path(set_dir).glob("seq*")):
        z = np.load(d / "seq.npz")
        Rc, tc, fx = z["R"], z["t"], float(z["fx"])
        truth = Truth(d, fk)
        for line in open(d / landmarks):
            f = json.loads(line)
            for side, S in (("l", "left"), ("r", "right")):
                L = f.get(f"{S}Hand2d")
                if not L:
                    continue
                C = {s: (truth.posed_points(s, f["t"]) - tc) @ Rc.T for s in "lr"}
                img = {s: project(C[s], fx) for s in "lr"}
                p = np.array(L)[[0, 9], :2].mean(0)
                owner = min("lr", key=lambda s: np.linalg.norm(p - img[s][[0, 9]].mean(0)))
                if owner != side:
                    continue
                M, T = image_points(L, side), truth_image(C[side], fx, side)
                _, Rm, lm_ = palm_frame(M)
                _, Rt, lt = palm_frame(T)
                Rt_mirror = Rt * np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]])  # palm and back swapped
                size = lt
                rows.append({"ang": rot_angle(Rm, Rt), "ang_mirror": rot_angle(Rm, Rt_mirror),
                             "xy": np.linalg.norm((M - M[0])[:, :2] - (T - T[0])[:, :2], axis=-1).mean() / size,
                             "z": np.abs((M - M[0])[:, 2] - (T - T[0])[:, 2]).mean() / size,
                             "pos": float(np.linalg.norm(M[0, :2] - T[0, :2]) / size),
                             "size_ratio": float(lm_ / lt)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", type=Path)
    ap.add_argument("--landmarks", default="landmarks.jsonl")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    rows = collect(a.set, a.landmarks)
    ang = np.array([r["ang"] for r in rows])
    mirrored = np.array([r["ang_mirror"] < r["ang"] - 30 for r in rows])
    out = {"n": len(rows), "orientation_deg_median": float(np.median(ang)), "orientation_deg_p90": float(np.percentile(ang, 90)),
           "mirrored": float(mirrored.mean()), "orientation_deg_median_unmirrored": float(np.median(ang[~mirrored])),
           "xy_err": float(np.median([r["xy"] for r in rows])), "z_err": float(np.median([r["z"] for r in rows])),
           "wrist_pos_err": float(np.median([r["pos"] for r in rows])),
           "size_ratio_median": float(np.median([r["size_ratio"] for r in rows])),
           "size_ratio_iqr": np.percentile([r["size_ratio"] for r in rows], [25, 75]).round(3).tolist()}
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
