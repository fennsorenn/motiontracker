"""MHR -> VRM finger conversion (mtcore.convert) replayed on the avatar reproduces MHR's fingers, measured
with the hand model's metric (mttrain.hands.evaluate): palm roll and distal bones (fixed 2026-11-12)."""

import numpy as np

from mtcore import mhr
from mtcore.convert import MhrToVrm
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm
from mttrain.hands.evaluate import angle_errors, vrm_points
from mttrain.hands.motion import HandMotion


def test_fingers_replayed_on_the_avatar_match_mhr():
    hm = HandMotion()
    fk = hm.fk
    rest = load_vrm(AVATARS / "AvatarSample_D.vrm")[2]
    conv = MhrToVrm(fk.jn, mhr.skeleton_state(fk.ch, np.zeros(len(fk.pn))), rest)
    rng = np.random.default_rng(0)
    Q = np.stack([hm.random_pose(rng) for _ in range(6)] + [hm.library[n] for n in ("fist", "ok", "splay")])
    mp = fk.full(Q, "l", base=fk.full(Q, "r"))  # same finger pose on both hands
    L = conv.locals(mhr.skeleton_state(fk.ch, mp))
    for i in range(len(Q)):
        loc = {b: L[b][i] for b in L}
        for side in "lr":
            err = angle_errors(vrm_points(loc, side, rest), fk.points(Q[i], side))
            assert err.max() < 0.5, (i, side, err.round(1))
