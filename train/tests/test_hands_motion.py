"""Procedural finger motion (mttrain.hands): limits, unobservable roll, determinism, thumb IK, smoothness."""

import numpy as np
import pytest

from mttrain.hands.motion import GESTURES, HandMotion
from mttrain.hands.skeleton import MP21, PARAMS, UNOBSERVABLE, HandFK, hand_frame


@pytest.fixture(scope="module")
def hm():
    return HandMotion()


def test_tables():
    assert len(PARAMS) == len(set(PARAMS)) == 27
    assert len(MP21) == 21


def test_right_hand_mirrors_left(hm):
    q = hm.random_pose(np.random.default_rng(0))
    fk: HandFK = hm.fk
    L = fk.points(q, "l")
    R = fk.points(q, "r")
    Lw, Rw = L - L[0], R - R[0]
    np.testing.assert_allclose(Lw * [-1, 1, 1], Rw, atol=1e-4)


def test_gesture_thumbs_reach_their_targets(hm):
    for name, g in GESTURES.items():
        q = hm.library[name]
        pts = hm.fk.points(q)
        th = g["thumb"]
        tgt = hm._to_hand(th[1:4]) if th[0] == "target" else pts[th[1]] + (np.asarray(th[2:5]) / 100) @ hm.R
        assert np.linalg.norm(pts[4] - tgt) < 0.015, name


def test_sequence_limits_roll_and_determinism(hm):
    a, la = hm.sequence(np.random.default_rng(3), 10.0, 30.0)
    b, lb = hm.sequence(np.random.default_rng(3), 10.0, 30.0)
    np.testing.assert_array_equal(a, b)
    assert la == lb and len(la) >= 2
    assert a.shape == (300, 27)
    assert (a >= hm.lo - 1e-9).all() and (a <= hm.hi + 1e-9).all()
    assert (a[:, [PARAMS.index(p) for p in UNOBSERVABLE]] == 0).all()
    # smooth: no joint faster than 12 rad/s (fast real finger motion peaks around 10)
    assert np.abs(np.diff(a, axis=0)).max() * 30 < 12


def test_curl_closes_the_hand(hm):
    """Fist fingertips end up nearer the wrist and on the palm side, open ones far along the hand."""
    for name, near in (("fist", True), ("open", False)):
        P = hm.fk.points(hm.library[name])
        w, R = hand_frame(hm.fk.points(np.zeros(27)))
        tips = (P[[8, 12, 16, 20]] - w) @ R.T
        assert (tips[:, 0] < 0.10).all() == near, name
