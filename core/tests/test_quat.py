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


def test_mat_to_quat_exact_near_180_degrees(rng):
    from mtcore.lifter_pose import mat_to_quat, quat_to_mat
    q = rng.normal(size=(2000, 4))
    q[:1000, 3] = rng.normal(scale=1e-7, size=1000)  # rotations within ~1e-5 deg of 180
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_to_mat(q)
    assert np.abs(quat_to_mat(mat_to_quat(R)) - R).max() < 1e-9
