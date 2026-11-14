"""The merger bundle in the runtime (needs the gitignored bundle in data/; skipped without it)."""

import json

import numpy as np
import pytest

from mtcore.paths import DATA

from mtservice.cli import DEFAULT_MERGER, MergerModel

LANDMARKS = DATA / "videos" / "seated-v1-full" / "face.landmarks.jsonl"
pytestmark = pytest.mark.skipif(not (DEFAULT_MERGER / "merger.json").exists() or not LANDMARKS.exists(),
                                reason="merger bundle or landmarks dump not in data/")


def frames(n):
    rows = [json.loads(line) for _, line in zip(range(n), open(LANDMARKS))]
    return [(r["t"] / 1000.0, {k: v if k == "blendshapes" or not isinstance(v, list) else np.asarray(v) for k, v in r.items()})
            for r in rows]


def test_merger_outputs_body_legs_fingers_root_and_up():
    m = MergerModel(DEFAULT_MERGER, head3d=True, fingers=True, legs=True)
    for t, obs in frames(30):
        bones, conf, present, root, up = m.step(t, obs)
    assert present and {"hips", "head", "leftHand", "leftUpperLeg", "rightFoot", "leftIndexProximal"} <= set(bones)
    for q in bones.values():
        assert np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-6
    assert 0.3 < root[2] < 5 and abs(np.linalg.norm(up) - 1) < 1e-6
    assert all(np.isfinite(c) and c > 0 for c in conf.values())  # exp(-log scale), as the lifter's /MT/Conf


def test_no_legs_no_fingers():
    m = MergerModel(DEFAULT_MERGER, head3d=True, fingers=False, legs=False)
    t, obs = frames(1)[0]
    bones = m.step(t, obs)[0]
    assert "hips" in bones and not any("Leg" in b or "Foot" in b or "Index" in b for b in bones)
