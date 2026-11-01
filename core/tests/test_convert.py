import numpy as np

from conftest import needs_avatars, needs_mhr
from mtcore.convert import MAP, VRM_CHILD, MhrToVrm, vrm_fk
from mtcore.mhr import FLEX_T, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm


@needs_mhr
@needs_avatars
def test_bone_directions_survive_conversion(mhr, rng):
    """Random MHR poses -> VRM local rotations -> VRM FK: every bone points where its MHR bone does."""
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn)))
    _, _, vrm_rest = load_vrm(AVATARS / "AvatarSample_D.vrm")
    conv = MhrToVrm(jn, rest, vrm_rest)
    pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
    worst = 0.0
    for _ in range(20):
        mp = np.zeros(len(pn), np.float32)
        mp[pose_idx] = rng.uniform(-0.6, 0.6, len(pose_idx))
        st = skeleton_state(ch, mp)
        P, _ = vrm_fk(conv.locals(st), vrm_rest)
        for b in conv.bones:
            c, jd = VRM_CHILD.get(b), MAP[b][1]
            if c not in P or jd is None or b == "hips":  # hips: see spikes/m3-mhr-vrm/test_roundtrip.py
                continue
            dv, dm = P[c] - P[b], st[jn.index(jd), :3] - st[jn.index(MAP[b][0]), :3]
            cos = np.dot(dv, dm) / np.linalg.norm(dv) / np.linalg.norm(dm)
            worst = max(worst, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    assert worst < 0.1
