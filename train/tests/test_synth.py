import numpy as np
import pytest

from mtcore.landmarks import MP33
from mtcore.paths import MHR_ASSETS
from mttrain.mpstyle import DEPTH_KEYS, Style
from mttrain.noise import NoiseConfig, observe, outside_distance

I = MP33.index


def lm(rng):
    W = rng.normal(size=(4, 33, 3)) * 0.3
    return W - 0.5 * (W[:, [I("left_hip")]] + W[:, [I("right_hip")]])


def dist(X, a, b):
    return np.linalg.norm(X[:, I(a)] - X[:, I(b)], axis=-1)


def test_identity_style_is_a_no_op():
    W = lm(np.random.default_rng(0))
    s = Style.identity()
    assert np.allclose(s.world(W), W) and np.allclose(s.placement(W), W)


def test_depth_flattening():
    rng = np.random.default_rng(3)
    W = lm(rng)
    W = W - 0.5 * (W[:, I("left_hip")] + W[:, I("right_hip")])[:, None]
    assert np.allclose(Style.identity().flatten(W), W)
    s = Style.calibrated()
    F = s.flatten(W)
    g, b = dict(zip(DEPTH_KEYS, s.depth_gain)), dict(zip(DEPTH_KEYS, s.depth_bias))
    for seg, (p, q) in {"upper_arm": ("left_shoulder", "left_elbow"), "forearm": ("right_elbow", "right_wrist"),
                        "ears": ("right_ear", "left_ear"), "hand": ("left_wrist", "left_index")}.items():
        d, e = W[:, I(q)] - W[:, I(p)], F[:, I(q)] - F[:, I(p)]
        L = np.linalg.norm(d, axis=-1)
        assert np.allclose(e[:, 2], np.clip(g[seg] * d[:, 2] + b[seg], -L, L))  # depth: gain x true + bias
        ok = np.linalg.norm(d[:, :2], axis=-1) > 0.3 * L  # not pointing at the camera (stretch cap)
        assert np.allclose(np.linalg.norm(e, axis=-1)[ok], L[ok])  # length kept
        cos = (e[:, :2] * d[:, :2]).sum(-1) / np.linalg.norm(e[:, :2], axis=-1) / np.linalg.norm(d[:, :2], axis=-1)
        assert np.allclose(cos[ok], 1.0)  # image-plane direction kept
    # legs below the hips: only the hip step translates them
    assert np.allclose(F[:, I("left_ankle")] - F[:, I("left_knee")], W[:, I("left_ankle")] - W[:, I("left_knee")])
    mid = lambda X: 0.5 * (X[:, I("left_hip")] + X[:, I("right_hip")])
    assert np.allclose(mid(F), 0.0)


def test_calibrated_style_factors():
    W = lm(np.random.default_rng(1))
    s = Style.calibrated()
    V = s.world(W)
    assert np.allclose(dist(V, "left_shoulder", "left_elbow") / dist(W, "left_shoulder", "left_elbow"), 0.77)
    assert np.allclose(dist(V, "left_shoulder", "right_shoulder"), dist(W, "left_shoulder", "right_shoulder"))
    P = s.placement(W)
    assert np.allclose(dist(P, "left_hip", "right_hip") / dist(W, "left_hip", "right_hip"), 1.4)
    mid = lambda X: 0.5 * (X[:, I("left_hip")] + X[:, I("right_hip")])
    assert np.allclose(mid(P), mid(W))


def test_quiet_noise_keeps_points_and_visibility_follows_frame():
    rng = np.random.default_rng(2)
    T = 50
    world = np.zeros((T, 33, 3))
    image = np.full((T, 33, 3), 0.5)
    image[:, I("left_wrist"), 1] = 1.6  # far below the frame
    quiet = NoiseConfig(jitter_scale=0.0, guess_sigma=0.0, vis_logit_sigma=0.0, glitch_rate=0.0, dropout_rate=0.0)
    W, Im, vis, present = observe(rng, world, image, np.full(T, 0.1), quiet)
    assert np.allclose(W, world) and np.allclose(Im, image) and present.all()
    assert (vis[:, I("nose")] > 0.95).all() and (vis[:, I("left_wrist")] < 0.05).all()
    assert outside_distance(image)[0, I("left_wrist")] == pytest.approx(0.6)


def test_noise_level_and_correlation():
    rng = np.random.default_rng(3)
    T = 4000
    cfg = NoiseConfig()
    W, _, _, _ = observe(rng, np.zeros((T, 33, 3)), np.full((T, 33, 3), 0.5), np.full(T, 0.1), cfg)
    x = W[:, I("left_shoulder"), 0]
    x = x[np.abs(x) < 0.05]  # drop glitches
    assert 0.0015 < x.std() < 0.006  # base 2.5 mm, inside the frame (edge curve x1.0-1.5)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(cfg.rho10, abs=0.05)


@pytest.mark.skipif(not (MHR_ASSETS / "lod1.fbx").exists(), reason="MHR assets missing")
def test_synthesize_shapes():
    from mtcore.landmarks import LandmarkModel
    from mtcore.mhr import load, skeleton_state
    from mttrain.synth import synthesize
    ch = load()
    pn = list(ch.parameter_transform.names)
    lmk = LandmarkModel(ch, skeleton_state(ch, np.zeros(len(pn))))
    mp = np.zeros((90, len(pn)), np.float32)
    s = synthesize(np.random.default_rng(4), ch, lmk, mp, fps=10.0)
    T = len(s["t"])
    assert 25 <= T <= 31 and s["world"].shape == (T, 33, 3) and s["vis"].shape == (T, 33)
    assert np.isfinite(s["world"]).all() and np.isfinite(s["image"]).all()


def test_visibility_lags_when_leaving_the_frame():
    rng = np.random.default_rng(5)
    T = 30
    image = np.full((T, 33, 3), 0.5)
    image[10:, I("left_wrist"), 1] = 1.6
    quiet = NoiseConfig(jitter_scale=0.0, guess_sigma=0.0, vis_logit_sigma=0.0, glitch_rate=0.0, dropout_rate=0.0)
    _, _, vis, _ = observe(rng, np.zeros((T, 33, 3)), image, np.full(T, 0.1), quiet)
    v = vis[:, I("left_wrist")]
    assert v[9] > 0.95 and 0.3 < v[11] < 0.95 and v[29] < 0.1  # decays over several frames
