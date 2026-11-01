import numpy as np

from mtcore.convert import q_between, qinv, qmul, qrot, twist


def unit(v):
    return v / np.linalg.norm(v)


def test_q_between_maps_direction(rng):
    for _ in range(50):
        a, b = unit(rng.normal(size=3)), unit(rng.normal(size=3))
        assert np.allclose(qrot(q_between(a, b), a), b, atol=1e-6)


def test_twist_is_rotation_about_axis(rng):
    for _ in range(50):
        q = unit(rng.normal(size=4))
        axis = unit(rng.normal(size=3))
        t = twist(q, axis)
        assert np.allclose(qrot(t, axis), axis, atol=1e-6)  # twist leaves its axis fixed
        swing = qmul(q, qinv(t))
        # swing moves the axis by the same amount q does, with no rotation about it left over
        assert np.allclose(qrot(swing, axis), qrot(q, axis), atol=1e-6)
