"""Metric checks on synthetic signals with known answers."""

import numpy as np
import pytest

from mteval import metrics as M
from mteval.session import PoseTrack, Segment, Session, auto_still_segments


def rot(axis, deg):
    """Quaternions [x,y,z,w] for rotations by `deg` (array) about `axis`."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(np.asarray(deg, float)) / 2
    return np.concatenate([np.sin(h)[:, None] * axis, np.cos(h)[:, None]], axis=1)


def track(t, **bones):
    return PoseTrack(np.asarray(t, float), {k: np.asarray(v, float) for k, v in bones.items()})


def smooth_motion(t_ms, delay_ms=0.0):
    """A bumpy, non-periodic angle trajectory (deg) for lag tests."""
    s = (t_ms - delay_ms) / 1000.0
    return 40 * np.sin(1.3 * s) + 25 * np.sin(3.1 * s + 1) + 15 * np.sin(5.7 * s + 2)


def test_angular_speed_constant_rate():
    t = np.arange(0, 2000, 100.0)
    sp = M.angular_speed(t, rot([0, 1, 0], 0.09 * t))  # 90 deg/s
    assert np.allclose(sp.speed, 90.0, atol=1e-6)


def test_angular_speed_skips_gaps():
    t = np.arange(0, 1000, 100.0)
    q = rot([1, 0, 0], 0.05 * t)
    q[4] = np.nan
    sp = M.angular_speed(t, q)
    assert len(sp.speed) == len(t) - 1 - 2  # both intervals touching the missing frame dropped


@pytest.mark.parametrize("delay", [0.0, 80.0, 250.0])
def test_xcorr_lag_recovers_delay_across_rates(delay):
    t_ref = np.arange(0, 20000, 100.0)  # 10 Hz "detector"
    t_out = np.arange(0, 20000, 1000 / 60)  # 60 Hz "rendered"
    ref = M.angular_speed(t_ref, rot([0, 0, 1], smooth_motion(t_ref)))
    out = M.angular_speed(t_out, rot([0, 0, 1], smooth_motion(t_out, delay)))
    r = M.xcorr_lag(ref, out)
    assert r["lag_ms"] == pytest.approx(delay, abs=15)
    assert r["corr"] > 0.9


def test_rest_metrics_scale_with_noise():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10000, 100.0)
    still = [Segment("still", 0, 10000)]
    results = []
    for sigma in (0.5, 2.0):
        q = rot([0, 1, 0], rng.normal(0, sigma, len(t)))
        tr = track(t, neck=q)
        results.append((M.rest_jitter(tr, still)["neck"], M.rest_noise(tr, still)["neck"]))
    (j1, n1), (j2, n2) = results
    assert j2 > 3 * j1 and n2 > 3 * n1
    # Independent noise with std sigma: rest noise (RMS deviation from local mean) is ~sigma.
    assert n2 == pytest.approx(2.0, rel=0.3)


def test_twitch_rate_counts_runs():
    t = np.arange(0, 60000, 100.0)  # one minute still
    ang = np.zeros(len(t))
    ang[100:102] = 40  # one twitch spanning two frames: up, hold, down
    ang[300] = 40  # a second, single-frame twitch
    tr = track(t, neck=rot([1, 0, 0], ang))
    rate = M.twitch_rate(tr, [Segment("still", 0, 60000)], threshold_deg_s=180)["neck"]
    # 40 deg in 100 ms = 400 deg/s. First twitch: up and down are separated by a slow interval,
    # so it counts twice; the single-frame twitch is one run (up then down back to back).
    assert rate == pytest.approx(3.0)


def test_presence():
    q = rot([1, 0, 0], np.zeros(4))
    q[1] = np.nan
    assert M.presence(track([0, 1, 2, 3], neck=q))["neck"] == 0.75


def test_rotation_error_constant_offset():
    t = np.arange(0, 2000, 100.0)
    base = smooth_motion(t)
    gt = track(t, neck=rot([0, 1, 0], base))
    pred = track(t, neck=rot([0, 1, 0], base + 10))
    e = M.rotation_error(pred, gt)["neck"]
    assert e["mean"] == pytest.approx(10.0, abs=1e-6) and e["coverage"] == 1.0


def test_onset_lag_labeled():
    t = np.arange(0, 20000, 1000 / 30)
    gt = track(t, neck=rot([0, 1, 0], smooth_motion(t)))
    pred = track(t, neck=rot([0, 1, 0], smooth_motion(t, 100)))
    assert M.onset_lag(pred, gt)["neck"]["lag_ms"] == pytest.approx(100, abs=15)


def test_peak_velocity_ratio_half_amplitude():
    t = np.arange(0, 20000, 1000 / 30)
    base = smooth_motion(t)
    gt = track(t, neck=rot([0, 1, 0], base))
    pred = track(t, neck=rot([0, 1, 0], 0.5 * base))
    r = M.peak_velocity_ratio(pred, gt)["neck"]
    assert r["peaks"] > 5
    assert r["median"] == pytest.approx(0.5, abs=0.05)


def _session_with_pose(t, wrist_x, left_hand_present):
    frames = []
    for k, tt in enumerate(t):
        pose = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
        pose[15] = [wrist_x[k], 0.3, 0.0, 1.0]
        f = {"t": float(tt), "pose": pose}
        if left_hand_present[k]:
            f["leftHand"] = [[0.0, 0.0, 0.0]] * 21
        frames.append(f)
    return Session("synthetic", None, frames)


def test_auto_still_segments():
    t = np.arange(0, 10000, 100.0)
    x = np.where((t > 3000) & (t < 6000), 0.0005 * (t - 3000), 0.0)  # moves 3 s - 6 s
    x[t >= 6000] = 1.5
    segs = auto_still_segments(_session_with_pose(t, x, np.ones(len(t), bool)))
    assert len(segs) == 2
    assert segs[0].t1 <= 3100 and segs[1].t0 >= 6000


def test_dropout_jump_sees_snap():
    t = np.arange(0, 4000, 100.0)
    present = t < 2000
    sess = _session_with_pose(t, np.zeros(len(t)), present)
    ang = np.where(t < 2000, 0.0, 30.0)  # the hand bone snaps 30 deg when the hand disappears
    tr = track(t, leftHand=rot([1, 0, 0], ang))
    r = M.dropout_jump(tr, sess)["leftHand"]
    assert r["transitions"] == 1
    assert r["mean_peak"] == pytest.approx(300.0)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # all-zero landmarks in lag metric
def test_rest_summary_per_still_segment(tmp_path):
    import json

    from mteval.report import evaluate, to_markdown
    from mteval.session import load_session
    from mteval.systems import SystemResult

    t = np.arange(0, 20000, 100.0)
    frames = [{"t": float(tt), "pose": [[0.0, 0.0, 0.0, 1.0]] * 33} for tt in t]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(json.dumps(f) for f in frames))
    segs = [
        {"label": "still", "variant": "arms-below-frame", "t0": 0, "t1": 10000},
        {"label": "still", "t0": 10000, "t1": 20000},
    ]
    (tmp_path / "session.json").write_text(json.dumps({"name": "x", "segments": segs}))
    sess = load_session(tmp_path)
    assert sess.name == tmp_path.name and sess.segments[0].variant == "arms-below-frame"

    rng = np.random.default_rng(1)
    sigma = np.where(t < 10000, 3.0, 0.3)  # noisy first rest segment, calm second
    res = SystemResult("sys", track(t, neck=rot([0, 1, 0], rng.normal(0, 1, len(t)) * sigma)), [])
    rep = evaluate(sess, [res])
    seg = rep["still_segment_summary"]["sys"]
    first, second = seg["arms-below-frame (0-10 s)"], seg["still 10-20 s"]
    assert first["torso_head"]["rest_noise"] > 5 * second["torso_head"]["rest_noise"]
    assert "Rest by still segment" in to_markdown(rep)
