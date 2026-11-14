"""Pool of procedural finger sequences for training (finger parameters of the canonical left hand at the
runtime's 10 Hz). Purely procedural (no recorded data).

    uv run python -m mttrain.hands.pool --split train --n 6000 [--seconds 30] [--seed 0]
    uv run python -m mttrain.hands.pool --split val --n 300 --seed 1

writes data/m4/pool/<split>.npz: q (N, T, 27) float32.
"""

import argparse
import time

import numpy as np

from mtcore.paths import DATA

from .motion import HandMotion

FPS = 10.0
POOL = DATA / "m4" / "pool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    hm = HandMotion()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    Q = np.stack([hm.sequence(rng, a.seconds, FPS)[0] for _ in range(a.n)]).astype(np.float32)
    POOL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POOL / f"{a.split}.npz", q=Q, fps=FPS)
    print(f"{a.split}: {Q.shape} in {time.time() - t0:.0f} s -> {POOL / f'{a.split}.npz'}")


if __name__ == "__main__":
    main()
