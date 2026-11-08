"""The torch generator against the numpy reference."""

import numpy as np
import torch

from mtcore.mhr import load, skeleton_state
from mttrain.gpugen.fk import MhrFK


def _qangle(a, b):
    d = np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(d, 0, 1)))


def test_fk_matches_pymomentum():
    ch = load()
    rng = np.random.default_rng(0)
    pn = list(ch.parameter_transform.names)
    mp = rng.normal(0, 0.3, (16, len(pn)))
    mp[:, [i for i, n in enumerate(pn) if n.startswith("scale_")]] *= 0.3
    ref = skeleton_state(ch, mp)
    out = MhrFK(ch)(torch.tensor(mp, dtype=torch.float32)).double().numpy()
    assert np.abs(out[..., :3] - ref[..., :3]).max() < 1e-2  # cm
    assert _qangle(out[..., 3:7], ref[..., 3:7]).max() < 0.05  # deg
    assert np.abs(out[..., 7] - ref[..., 7]).max() < 1e-4
