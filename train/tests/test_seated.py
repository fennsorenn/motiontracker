import numpy as np
import pytest

from mtcore.paths import MHR_ASSETS
from mttrain.seated import REST, min_jerk, rest_weights

needs_mhr = pytest.mark.skipif(not (MHR_ASSETS / "lod1.fbx").exists(), reason="MHR assets missing")


def test_min_jerk_and_rest_weights():
    assert min_jerk(np.array([0.0, 0.5, 1.0])).tolist() == [0.0, 0.5, 1.0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        w, names = rest_weights(rng, 900, 30.0, p_rest=1.0)
        assert ((w >= 0) & (w <= 1)).all()
        assert set(names[w > 0]) <= set(REST)
        assert np.abs(np.diff(w)).max() < 0.2  # no pops (fastest ramp 0.4 s: peak 0.16 per frame)


@needs_mhr
def test_seated_geometry_and_limits():
    from mtcore.mhr import limits, load, positions
    from mttrain.seated import seat
    ch = load()
    pn = list(ch.parameter_transform.names)
    jn = list(ch.skeleton.joint_names)
    lim = limits()
    rng = np.random.default_rng(1)
    mp = np.zeros((300, len(pn)), np.float32)
    mp[:, pn.index("root_ry")] = np.linspace(0, 3, 300)  # a turning, standing clip
    ms = seat(rng, pn, mp, limits=lim)
    X = positions(ch, ms)
    J = lambda n: X[:, jn.index(n)]
    for s in "lr":
        assert (J(f"{s}_lowleg")[:, 2] - J(f"{s}_upleg")[:, 2] > 25).all()  # thighs forward
        assert (J(f"{s}_upleg")[:, 1] - J(f"{s}_foot")[:, 1] > 25).all()  # feet down
    assert np.abs(ms[:, pn.index("root_ry")]).max() < 0.6  # facing the desk, not following the walk
    for i, n in enumerate(pn):
        lo, hi = lim.get(n, (0, 0))
        if hi > lo and not n.startswith("scale_"):
            assert lo - 1e-6 <= ms[:, i].min() and ms[:, i].max() <= hi + 1e-6, n
