"""Contrast configurations for the probe: one seated body, one camera, one face target, the hand at
several gaps from the face along the face normal (contact .. clear hover). Each configuration is a
short "sequence" the Blender script renders frame by frame (one frame per gap).

    cd train && uv run python ../spikes/mp-depth-probe/make_configs.py --n 80 [--seed 7]

Writes data/spikes/mp-depth/configs/cNNN/seq.npz (render input: verts, faces, R, t, fx, skin, wall)
and meta.npz (gaps cm, true camera-frame landmarks (G, 33, 3) m, pose2d projections (G, 33, 3),
face point, side, contact type). Bodies from held-out subjects (CMU-derived: local only).
"""

import argparse
from pathlib import Path

import numpy as np

from mtcore.landmarks import MP33, LandmarkModel
from mtcore.mhr import limits, load, skeleton_state
from mtcore.paths import DATA
from mttrain import choreo
from mttrain.camera import sample_desk_camera
from mttrain.dataset import FITS, clip_table
from mttrain.seated import seat

GAPS = (0.5, 4.0, 8.0, 14.0)  # cm in front of the face surface, along its outward normal
OUT = DATA / "spikes" / "mp-depth" / "configs"


def targets(X, name, side, palm, gap, up):
    """choreo's face constraints with an explicit gap (choreo uses gap 1 cm for contacts)."""
    F, n = choreo._face_point(X, name, side)
    if palm:
        return [(f"{side}_middle1", F + n * (gap + 1.0), 1.0, True), (f"{side}_wrist", F + n * (gap + 2.0) - up * 7, 0.2, False),
                (f"{side}_middle_null", F + n * (gap + 0.5) + up * 7, 0.1, False)]
    return [(f"{side}_middle_null", F + n * (gap - 0.2), 1.0, True), (f"{side}_wrist", F + n * (gap + 6) - up * 9, 0.05, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ch = load()
    pn = list(ch.parameter_transform.names)
    jn = list(ch.skeleton.joint_names)
    rest = skeleton_state(ch, np.zeros(len(pn)))
    lm = LandmarkModel(ch, rest)
    lim = limits()
    P = {n: i for i, n in enumerate(pn)}
    rows = clip_table("val")
    faces = np.asarray(ch.mesh.faces, dtype=np.int32)
    k = 0
    while k < a.n:
        key = rows[rng.integers(len(rows))][0]
        mps = np.load(FITS / f"{key}.npz")["mp"]
        base = mps[rng.integers(len(mps))].astype(np.float64)
        body = np.where([n.startswith("scale_") for n in pn], base, 0.0)
        m0 = seat(rng, pn, body[None], 30.0, lim, force_rest=tuple(rng.choice(["lap", "desk"], 2)))[0]
        m0[[P[h] for h in choreo.HEAD_KEYS]] += [rng.uniform(*v) for v in choreo.HEAD_KEYS.values()]
        X = lm(skeleton_state(ch, m0)[None])[0]
        _, _, up, _ = choreo._body_frame(X)
        name = choreo.FACE_POINTS[rng.integers(len(choreo.FACE_POINTS))]
        side = "l" if rng.random() < 0.5 else "r"
        palm = rng.random() < 0.4
        sols, ok, prev = [], True, m0
        for gap in GAPS:
            tg = targets(X, name, side, palm, gap, up)
            sol = choreo._solve(ch, pn, jn, prev, (side,), tg)
            prim = [(j, p) for j, p, _, pr in tg if pr]
            res = np.linalg.norm(choreo._joint_pos(ch, sol, [j for j, _ in prim], jn) - np.stack([p for _, p in prim]), axis=-1)
            if res.max() > choreo.MAX_RESIDUAL_CM:
                ok = False
                break
            sols.append(sol)
            prev = sol
        if not ok:
            continue
        mp = np.stack(sols)
        states = skeleton_state(ch, mp)
        Xs = lm(states) / 100.0
        cam = sample_desk_camera(rng, Xs)
        im, _ = cam.project(Xs)
        if not ((im[..., :2] > 0.02) & (im[..., :2] < 0.98)).all(axis=-1)[:, [MP33.index("nose"), MP33.index(f"{'left' if side == 'l' else 'right'}_wrist")]].all():
            continue  # hand or face out of frame: resample
        d = OUT / f"c{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        verts = np.asarray(ch.skin_points(states.astype(np.float32))) / 100.0
        skin = np.clip(np.array([0.80, 0.60, 0.48]) * rng.uniform(0.55, 1.1) + rng.normal(0, 0.03, 3), 0.05, 1.0)
        np.savez_compressed(d / "seq.npz", verts=verts.astype(np.float32), faces=faces, R=cam.R, t=cam.t, fx=cam.fx,
                            mp=mp.astype(np.float32), fps=30.0, skin=skin, wall=rng.uniform(0.45, 0.9, 3), clip=key)
        np.savez_compressed(d / "meta.npz", gaps=np.array(GAPS), cam_xyz=cam.to_camera(Xs), pose2d=im, face_point=name,
                            side=side, palm=palm, fx=cam.fx, clip=key)
        print(f"c{k:03d}: body {key}, {name} {side} {'palm' if palm else 'fingertip'}, fov {np.degrees(2 * np.arctan(0.5 / cam.fx)):.0f}",
              flush=True)
        k += 1


if __name__ == "__main__":
    main()
