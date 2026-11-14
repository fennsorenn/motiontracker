import numpy as np

from mtcore.features import StreamFeatures, frame_features, history_indices, model_input


def _sequence(rng, T=120):
    """Irregular 10 FPS-ish times with dropped frames, absent poses and low-visibility shoulders."""
    dt = rng.choice([0.1, 0.1, 0.1, 0.2, 0.3, 0.09, 0.11], size=T)
    times = np.cumsum(dt)
    world = rng.normal(0, 0.3, (T, 33, 3))
    world[:, 11], world[:, 12] = [-0.17, 0, 0], [0.17, 0, 0]
    world[:, 11:13] += rng.normal(0, 0.02, (T, 2, 3))
    image = rng.uniform(0, 1, (T, 33, 3))
    vis = rng.uniform(0.3, 1, (T, 33))
    present = rng.uniform(size=T) > 0.1
    present[:3] = False  # starts without a pose: default scale
    vis[40:70, 11] = 0.2  # no usable shoulder width for longer than the scale window
    return times, world, image, vis, present


def test_stream_matches_batch():
    rng = np.random.default_rng(0)
    for head3d in (True, False):
        times, world, image, vis, present = _sequence(rng)
        feats = frame_features(times, world, image, vis, present, head3d)
        idx, age = history_indices(times, np.arange(len(times)))
        batch = model_input(feats, idx, age)
        s = StreamFeatures(head3d)
        stream = np.stack([s.push(times[i], world[i], image[i], vis[i], bool(present[i])) for i in range(len(times))])
        assert stream.dtype == batch.dtype
        np.testing.assert_allclose(stream, batch, rtol=0, atol=1e-6)
        assert len(s.frames) < 40  # history is pruned
