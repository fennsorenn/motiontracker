"""Pool of posed two-hand sequences for the pair hand model (v2): arms and fingers, with the body landmarks
the GPU generator needs to place a camera.

    MT_FITS=fits-v3 uv run python -m mttrain.hands.pool2 --split train --n 1500 [--seconds 20] [--seed 0]
    MT_FITS=fits-v3 uv run python -m mttrain.hands.pool2 --split val --n 100 --seed 1

Each sequence: a body of the split's subjects (proportions only), seated, arms and fingers from either the
two-hand interaction choreography (`pairs`, share P_PAIRS) or hands-in-use arms with independent procedural
fingers (`stage` + `motion`; arms rest now and then, in the lap, on the desk or hanging, with relaxed fingers), at 30 fps, kept at the runtime's 10 Hz. Writes data/m4/pool2/<split>.npz:
hands (N, T, 2, 21, 3) MediaPipe's hand points per side (l, r) in MHR world metres, body (N, T, 33, 3) the
MediaPipe-style body landmarks (camera placement), q (N, T, 2, 27) finger parameters, kind (N,) 0 = stage,
1 = pairs. Every sequence is collision-relaxed (`relax`); ones whose hands still overlap more than
FINAL_OVERLAP_CM, or with hands or forearms inside the body mesh in more than MAX_CLIP_FRAMES of the frames,
are drawn again (5 tries, then skipped). CMU-derived (body proportions): local only. CPU (pymomentum IK): run under tools/guarded-run.sh.
"""

import argparse
import time

import numpy as np

from mtcore.landmarks import LandmarkModel
from mtcore.mhr import limits, load, positions, skeleton_state
from mtcore.paths import DATA

from ..dataset import FITS, clip_table
from . import stage
from .motion import HandMotion
from .pairs import FINAL_OVERLAP_CM, PairChoreo
from .skeleton import HandFK

POOL2 = DATA / "m4" / "pool2"
P_PAIRS = 0.6
MAX_CLIP_FRAMES = 0.02  # share of frames with a hand or forearm inside the body mesh (pairs.mesh_clip_frames)
P_REST = 0.18  # stage sequences: chance per arm move of resting a while (lap, desk, hanging; v3 off-screen)
FPS, OUT_FPS = 30.0, 10.0


def stage_sequence(rng, ch, lm, pn, base, T, fps, lim, hm, pc):
    """Hands-in-use arms (`stage`, v3: rest episodes in the lap, on the desk or hanging, resting arms lifted out
    of the body, key poses clear of the body mesh) with procedural fingers that relax while the arm rests.
    -> mp (T, P), {side: finger key labels}."""
    mp, rest_w = stage.generate(rng, ch, lm, pn, base, T, fps, lim, p_rest=P_REST, rests=("lap", "desk", "hanging"),
                                with_rest=True, fix_rest=pc.fix_rest, clear=lambda m: pc.mesh_clear(m[None]))
    labels = {}
    for side in "lr":  # fingers relax while the arm rests (often out of the camera's view)
        q, labels[side] = hm.sequence(rng, T / fps, fps)
        wr = rest_w[side][:, None]
        q_rest = hm.relaxed_pose(rng) + hm._idle(rng, np.arange(T) / fps)
        mp[:, pc.fing[side]] = np.clip((1 - wr) * q + wr * q_rest, hm.lo, hm.hi)
    return mp, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="output file stem (default: the split); shards are merged with --merge")
    ap.add_argument("--merge", nargs="+", help="merge these stems' files into --name (no generation)")
    ap.add_argument("--dir", default=str(POOL2), help="pool directory (default data/m4/pool2)")
    a = ap.parse_args()
    from pathlib import Path
    pool = Path(a.dir)
    if a.merge:
        parts = [np.load(pool / f"{m}.npz") for m in a.merge]
        np.savez_compressed(pool / f"{a.name}.npz", **{k: np.concatenate([p[k] for p in parts]) for k in ("hands", "body", "q", "kind")},
                            fps=OUT_FPS)
        print(f"merged {a.merge} -> {a.name}: {sum(len(p['kind']) for p in parts)} sequences")
        return
    rng = np.random.default_rng(a.seed)
    ch = load()
    pn = list(ch.parameter_transform.names)
    lim = limits()
    lm = LandmarkModel(ch, skeleton_state(ch, np.zeros(len(pn))))
    fk = HandFK(ch)
    hm = HandMotion(fk)
    pc = PairChoreo(ch, lm, pn, lim, hm)
    rows = clip_table(a.split)
    keys = [k for k, _ in rows]
    w = np.array([wt for _, wt in rows])
    T = int(a.seconds * FPS)
    step = int(FPS / OUT_FPS)
    H, Bd, Q, K = [], [], [], []
    skipped = 0
    t0 = time.time()
    for i in range(a.n):
        key = keys[rng.choice(len(keys), p=w / w.sum())]
        clip = np.load(FITS / f"{key}.npz")["mp"]
        base = clip[rng.integers(len(clip))].astype(np.float64)
        kind = int(rng.random() < P_PAIRS)
        for _ in range(5):  # sequences whose hands still overlap after relaxation are drawn again
            if kind:
                mp, _ = pc.generate(rng, base, T, FPS)
                left = pc.last_overlap
            else:
                mp, _ = stage_sequence(rng, ch, lm, pn, base, T, FPS, lim, hm, pc)
                mp, _, after = pc.relaxer(mp)
                left = float(after.max())
            pc.mesh.set_pose(mp[0])
            clip = float(pc.mesh_clip_frames(mp[::3]).mean())  # hands or forearms inside the body (mesh)
            if left <= FINAL_OVERLAP_CM and clip <= MAX_CLIP_FRAMES:
                break
        else:
            skipped += 1
            continue
        mp = mp[::step]
        pos = positions(ch, mp.astype(np.float32)) / 100.0
        H.append(np.stack([pos[:, fk.mp21[s]] for s in "lr"], 1).astype(np.float16))
        Bd.append((lm(skeleton_state(ch, mp)) / 100.0).astype(np.float16))
        Q.append(np.stack([mp[:, fk.idx[s]] for s in "lr"], 1).astype(np.float16))
        K.append(kind)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{a.n} sequences, {time.time() - t0:.0f} s", flush=True)
    pool.mkdir(parents=True, exist_ok=True)
    name = a.name or a.split
    np.savez_compressed(pool / f"{name}.npz", hands=np.stack(H), body=np.stack(Bd), q=np.stack(Q),
                        kind=np.array(K, np.int8), fps=OUT_FPS)
    print(f"{name}: {len(H)} sequences ({skipped} skipped: hands overlapping or inside the body) in {time.time() - t0:.0f} s"
          f" -> {pool / f'{name}.npz'}")


if __name__ == "__main__":
    main()
