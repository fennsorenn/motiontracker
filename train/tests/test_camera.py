import numpy as np

from mtcore.landmarks import MP33
from mttrain.camera import look_at, sample_desk_camera, Camera


def person(facing=np.array([0.0, 0.0, 1.0])):
    """Crude 33-landmark person in MHR metres facing `facing` (+x = their left when facing +z)."""
    X = np.zeros((33, 3))
    X[:, 1] = 1.2
    for k, n in enumerate(MP33):
        side = 1 if n.startswith("left") or n.endswith("left") else -1 if ("right" in n) else 0
        X[k, 0] = 0.15 * side
        if "hip" in n:
            X[k, 1] = 0.9
        if n in ("nose",):
            X[k] = [0, 1.55, 0.1]
    return X


def test_left_is_image_right_and_axes():
    X = person()[None]
    cam = Camera(R=look_at(np.array([0, 1.4, 0.8]), np.array([0, 1.3, 0])), t=np.array([0, 1.4, 0.8]), fx=0.6)
    p2, front = cam.project(X)
    assert front.all()
    assert p2[0, MP33.index("left_shoulder"), 0] > p2[0, MP33.index("right_shoulder"), 0]  # as in sessions
    W = cam.world_landmarks(X)[0]
    assert W[MP33.index("left_shoulder"), 0] > 0  # x image-right
    assert W[MP33.index("nose"), 1] < 0  # y down: nose above hips
    assert W[MP33.index("nose"), 2] < 0  # z away from camera: nose closer
    assert np.allclose(0.5 * (W[MP33.index("left_hip")] + W[MP33.index("right_hip")]), 0)


def test_sampled_camera_sees_the_chest():
    rng = np.random.default_rng(0)
    X = np.repeat(person()[None], 5, 0)
    for _ in range(20):
        cam = sample_desk_camera(rng, X)
        p2, front = cam.project(X)
        sh = p2[0, [MP33.index("left_shoulder"), MP33.index("right_shoulder")]]
        assert front.all() and (sh[:, :2] > -0.2).all() and (sh[:, :2] < 1.2).all()
        assert sh[0, 0] > sh[1, 0]
