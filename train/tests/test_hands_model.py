"""Hand model pieces: torch canonical form = mtcore's, observer shapes and statistics, zero model = rest."""

import json

import numpy as np
import torch

from mtcore.hand_features import canonical
from mttrain.hands.model import HISTORY, HandNet, frame_input
from mttrain.hands.observe import Observer, canonical_t


def test_canonical_torch_matches_numpy():
    P = np.random.default_rng(0).normal(size=(5, 21, 3))
    np.testing.assert_allclose(canonical_t(torch.tensor(P)).numpy(), canonical(P, "l"), atol=1e-9)
    # right hands are mirrored left hands
    np.testing.assert_allclose(canonical(P * [-1, 1, 1], "r"), canonical(P, "l"), atol=1e-12)


def test_observer(tmp_path):
    rng = np.random.default_rng(1)
    cov = np.diag(np.full(63, 0.01))
    calib = {"fit": {"a": [0.0] * 63, "b": [0.6] * 63, "cov": cov.tolist(), "lag1": 0.7, "flipped": 0.01}}
    (tmp_path / "c.json").write_text(json.dumps(calib))
    obs = Observer(tmp_path / "c.json", "cpu")
    T = canonical_t(torch.tensor(rng.normal(size=(64, 8, 21, 3)), dtype=torch.float32))
    o, lost = obs(T, torch.Generator().manual_seed(0))
    assert o.shape == T.shape and lost.shape == (64, 8) and lost.dtype == torch.bool
    assert torch.allclose(o[..., 0, :], torch.zeros(3), atol=1e-5)  # wrist stays at the origin


def test_zero_model_is_rest():
    x = frame_input(torch.zeros(2, HISTORY, 21, 3), torch.zeros(2, HISTORY, dtype=torch.bool))
    pred, log_s = HandNet()(x)
    assert torch.all(pred == 0) and torch.all(log_s == 0)
