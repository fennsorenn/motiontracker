"""Pool of posed two-hand sequences for the pair hand model (v2): arms and fingers, with the body landmarks
the GPU generator needs to place a camera.

    MT_FITS=fits-v3 uv run python -m mttrain.hands.pool2 --split train --n 1500 [--seconds 20] [--seed 0]
    MT_FITS=fits-v3 uv run python -m mttrain.hands.pool2 --split val --n 100 --seed 1

Each sequence: a body of the split's subjects (proportions only), seated, arms and fingers from either the
two-hand interaction choreography (`pairs`, share P_PAIRS) or hands-in-use arms with independent procedural
fingers (`stage` + `motion`), at 30 fps, kept at the runtime's 10 Hz. Writes data/m4/pool2/<split>.npz:
hands (N, T, 2, 21, 3) MediaPipe's hand points per side (l, r) in MHR world metres, body (N, T, 33, 3) the
MediaPipe-style body landmarks (camera placement), q (N, T, 2, 27) finger parameters, kind (N,) 0 = stage,
1 = pairs. CMU-derived (body proportions): local only. CPU (pymomentum IK): run under tools/guarded-run.sh.
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
from .pairs import PairChoreo
from .skeleton import HandFK

POOL2 = DATA / "m4" / "pool2"
P_PAIRS = 0.6
FPS, OUT_FPS = 30.0, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
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
    t0 = time.time()
    for i in range(a.n):
        key = keys[rng.choice(len(keys), p=w / w.sum())]
        clip = np.load(FITS / f"{key}.npz")["mp"]
        base = clip[rng.integers(len(clip))].astype(np.float64)
        if rng.random() < P_PAIRS:
            mp, _ = pc.generate(rng, base, T, FPS)
            kind = 1
        else:
            mp = stage.generate(rng, ch, lm, pn, base, T, FPS, lim)
            for side in "lr":
                mp[:, fk.idx[side]] = hm.sequence(rng, T / FPS, FPS)[0]
            kind = 0
        mp = mp[::step]
        pos = positions(ch, mp.astype(np.float32)) / 100.0
        H.append(np.stack([pos[:, fk.mp21[s]] for s in "lr"], 1).astype(np.float16))
        Bd.append((lm(skeleton_state(ch, mp)) / 100.0).astype(np.float16))
        Q.append(np.stack([mp[:, fk.idx[s]] for s in "lr"], 1).astype(np.float16))
        K.append(kind)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{a.n} sequences, {time.time() - t0:.0f} s", flush=True)
    POOL2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POOL2 / f"{a.split}.npz", hands=np.stack(H), body=np.stack(Bd), q=np.stack(Q),
                        kind=np.array(K, np.int8), fps=OUT_FPS)
    print(f"{a.split}: {len(H)} sequences in {time.time() - t0:.0f} s -> {POOL2 / f'{a.split}.npz'}")


if __name__ == "__main__":
    main()
