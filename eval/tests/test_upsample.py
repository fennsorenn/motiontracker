import numpy as np

from mteval.quat import angle_deg
from mteval.session import PoseTrack
from mteval.upsample import glide


def _rot_x(deg):
    h = np.radians(deg) / 2
    return np.array([np.sin(h), 0.0, 0.0, np.cos(h)])


def test_constant_pose_unchanged_and_ticks_at_60hz():
    times = np.arange(0, 1000.1, 100.0)
    q = _rot_x(30)
    out = glide(PoseTrack(times, {"neck": np.tile(q, (len(times), 1))}))
    assert len(out.times) == 61
    np.testing.assert_allclose(np.diff(out.times), 1000 / 60)
    np.testing.assert_allclose(angle_deg(out.bones["neck"], q), 0, atol=1e-5)


def test_step_glides_over_one_interval_then_holds():
    times = np.arange(0, 2000.1, 100.0)
    q = np.array([_rot_x(0 if t < 1000 else 40) for t in times])
    out = glide(PoseTrack(times, {"neck": q}))
    ang = angle_deg(out.bones["neck"], _rot_x(0))
    t = out.times
    assert np.all(ang[t < 1000] < 1e-5)  # causal: nothing moves before the new pose arrives
    assert np.all(np.diff(ang) >= -1e-6)  # monotonic, no overshoot
    mid = np.argmin(np.abs(t - 1050))
    assert 15 < ang[mid] < 30  # roughly halfway after half an interval
    assert np.all(np.abs(ang[t >= 1100] - 40) < 1e-4)  # arrived within one interval


def test_absent_bone_stays_absent():
    times = np.arange(0, 500.1, 100.0)
    q = np.tile(_rot_x(10), (len(times), 1))
    q[2] = np.nan
    out = glide(PoseTrack(times, {"leftHand": q}))
    gap = (out.times >= 200) & (out.times < 300)
    assert np.isnan(out.bones["leftHand"][gap]).all()
    assert not np.isnan(out.bones["leftHand"][out.times >= 300]).any()
