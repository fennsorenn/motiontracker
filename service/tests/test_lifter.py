"""Needs a local bundle (mttrain.export) and a recorded session; skipped otherwise (both are gitignored)."""

import json

import numpy as np
import pytest

from mtcore.paths import REPO_ROOT
from mtservice.cli import DEFAULT_BUNDLE
from mtservice.lifter import LiveLifter

SESSION = REPO_ROOT / "sessions" / "2026-11-01T15-29-11-session" / "landmarks.jsonl"
REFERENCE = DEFAULT_BUNDLE.parent / "2026-11-01T15-29-11-session.poses.jsonl"  # mttrain.infer output


@pytest.mark.skipif(not (DEFAULT_BUNDLE / "model.onnx").exists() or not SESSION.exists() or not REFERENCE.exists(),
                    reason="needs the local m1-v10 bundle, session and offline poses")
def test_streaming_matches_offline_inference():
    lifter = LiveLifter(DEFAULT_BUNDLE)
    ref = [json.loads(l) for l in REFERENCE.read_text().splitlines()]
    worst = 0.0
    for i, line in enumerate(SESSION.read_text().splitlines()[:300]):
        f = json.loads(line)
        obs = {}
        if f.get("pose") and f.get("pose2d"):
            obs = {"pose": np.array(f["pose"]), "pose2d": np.array(f["pose2d"])}
        bones, conf, present = lifter.step(f["t"] / 1000.0, obs)
        for b, q in bones.items():
            worst = max(worst, float(np.abs(np.abs(np.dot(q, ref[i]["bones"][b])) - 1)))
        assert set(conf) == set(lifter.joints)
    assert worst < 1e-5  # |dot| within 1e-5 of 1: under about 0.5 deg even at the worst frame
