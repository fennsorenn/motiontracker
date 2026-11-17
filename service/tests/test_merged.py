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


def test_retarget_switch():
    from mtcore.paths import AVATARS
    vrm = AVATARS / "AvatarSample_D.vrm"
    if not vrm.exists() or not (DEFAULT_MERGER / "retarget.npz").exists():
        pytest.skip("avatar or retarget.npz not available")
    m = MergerModel(DEFAULT_MERGER, head3d=True, fingers=True, legs=False)
    m.set_avatar_vrm(vrm)
    assert m.retarget is not None and not m.retarget_on and not m.retarget_active  # set up, off by default
    seq = frames(20)

    def run(on):
        m.reset()
        m.retarget_on = on
        for t, obs in seq:
            bones = m.step(t, obs)[0]
        return bones

    off, on = run(False), run(True)
    arms = {f"{s}{b}" for s in ("left", "right") for b in ("UpperArm", "LowerArm", "Hand")}
    for b in off:
        same = np.allclose(off[b], on[b], atol=1e-9)
        assert same or b in arms, b  # only the arm chain changes
    assert any(not np.allclose(off[b], on[b], atol=1e-6) for b in arms)
    m.set_avatar(None)  # rest positions only: nothing to retarget to
    m.retarget_on = True
    assert not m.retarget_active
