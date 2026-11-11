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
    F = s.flatten(W)  # at the reference distance: the calibrated table
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
    # relaxed fully: no flattening at any distance; closer cameras flatten the forearm more
    s.depth_relax = 1.0
    assert np.allclose(s.flatten(W, 0.5), W)
    s.depth_relax = 0.0
    k = DEPTH_KEYS.index("forearm")
    assert s.depth_at(0.6)[0][k] < s.depth_at(1.1)[0][k] < s.depth_at(1.5)[0][k]


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


def _head_vs_chest(X):
    """Head-vs-chest rotation angle (deg) from landmarks: chest frame from shoulders and hips, head frame
    from ears, eyes and mouth (as mttrain.headcal)."""
    def fr(xa, up):
        x = xa / np.linalg.norm(xa, axis=-1, keepdims=True)
        z = np.cross(x, up); z /= np.linalg.norm(z, axis=-1, keepdims=True)
        return np.stack([x, np.cross(z, x), z], -1)
    C = fr(X[:, I("left_shoulder")] - X[:, I("right_shoulder")],
           X[:, [I("left_shoulder"), I("right_shoulder")]].mean(1) - X[:, [I("left_hip"), I("right_hip")]].mean(1))
    H = fr(X[:, I("left_ear")] - X[:, I("right_ear")],
           X[:, [I("left_eye"), I("right_eye")]].mean(1) - X[:, [I("mouth_left"), I("mouth_right")]].mean(1))
    return np.einsum("tji,tjk->tik", C, H)


def test_rotvec_round_trip():
    from mttrain.mpstyle import mat_to_rotvec, rotvec_to_mat
    v = np.random.default_rng(0).normal(0, 0.8, (50, 3))
    assert np.abs(mat_to_rotvec(rotvec_to_mat(v)) - v).max() < 1e-9


@pytest.mark.skipif(not (MHR_ASSETS / "lod1.fbx").exists(), reason="MHR assets missing")
def test_head_lock_gain():
    """Gain 1 without noise keeps the landmarks; gain 0 makes the face rigid with the chest (head-vs-chest
    rotation from landmarks as at rest) and leaves every other point alone."""
    from mtcore.landmarks import LandmarkModel
    from mtcore.mhr import load, skeleton_state
    from mttrain.mpstyle import FACE, HeadRig, head_lock
    ch = load()
    pn, jn = list(ch.parameter_transform.names), list(ch.skeleton.joint_names)
    rest = skeleton_state(ch, np.zeros(len(pn)))
    lmod = LandmarkModel(ch, rest)
    rig = HeadRig(jn, rest, lmod)
    rng = np.random.default_rng(2)
    mp = np.zeros((6, len(pn)))
    for k in ("neck_twist", "neck_lean", "neck_bend", "head_twist", "head_lean", "head_bend", "spine_twist0",
              "spine_bend1", "spine_lean0", "l_clavicle_rz", "root_ry", "root_rx"):
        mp[:, pn.index(k)] = rng.uniform(-0.3, 0.3, 6)
    st = skeleton_state(ch, mp)
    X = lmod(st) / 100.0
    delta, pivot = rig(st, X)
    d0, _ = rig(rest[None], lmod(rest[None]) / 100.0)
    assert np.abs(d0 - np.eye(3)).max() < 1e-5
    zero = np.zeros((6, 3))
    assert np.abs(head_lock(X, delta, pivot, 1.0, zero) - X).max() < 1e-6  # float32 skeleton states
    L = head_lock(X, delta, pivot, 0.0, zero)
    other = [k for k in range(33) if k not in FACE]
    assert np.abs(L[:, other] - X[:, other]).max() == 0
    r0 = _head_vs_chest(lmod(rest[None]) / 100.0)[0]
    ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))
    assert max(ang(r0, r) for r in _head_vs_chest(L)) < 0.01
    assert max(ang(r0, r) for r in _head_vs_chest(X)) > 10


def test_head3d_mask():
    from mttrain.features import HEAD3D_DIMS, frame_features
    rng = np.random.default_rng(0)
    T = 5
    world, image = rng.normal(0, 0.3, (T, 33, 3)), rng.uniform(0, 1, (T, 33, 3))
    vis, present = np.ones((T, 33)), np.ones(T, bool)
    full = frame_features(np.arange(T) * 0.1, world, image, vis, present)
    masked = frame_features(np.arange(T) * 0.1, world, image, vis, present, head3d=False)
    keep = [k for k in range(full.shape[1]) if k not in HEAD3D_DIMS]
    assert (masked[:, HEAD3D_DIMS] == 0).all() and np.array_equal(masked[:, keep], full[:, keep])
    assert len(HEAD3D_DIMS) == 44 and (full[:, HEAD3D_DIMS] != 0).any()


def test_hand_lock():
    """Gain 1 without noise keeps the landmarks; gain 0 straightens the hand onto the forearm direction
    (the bend's swing removed) and only moves the hand points."""
    from mttrain.mpstyle import HAND_SIDES, hand_lock
    rng = np.random.default_rng(5)
    X = rng.normal(0, 0.3, (7, 33, 3))
    zero = np.zeros((7, 2, 3))
    assert np.abs(hand_lock(X, 1.0, zero) - X).max() < 1e-9
    L = hand_lock(X, 0.0, zero)
    moved = sorted(p for _, _, pts in HAND_SIDES for p in pts)
    other = [k for k in range(33) if k not in moved]
    assert np.abs(L[:, other] - X[:, other]).max() == 0
    for el, wr, pts in HAND_SIDES:
        f = L[:, wr] - L[:, el]
        h = 0.5 * (L[:, pts[0]] + L[:, pts[1]]) - L[:, wr]
        cos = (f * h).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(h, axis=1)
        assert (cos > 1 - 1e-9).all()
        # distances to the wrist are kept
        assert np.allclose(np.linalg.norm(L[:, pts] - L[:, [wr]], axis=-1), np.linalg.norm(X[:, pts] - X[:, [wr]], axis=-1))
