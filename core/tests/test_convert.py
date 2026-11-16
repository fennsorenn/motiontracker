import numpy as np

from conftest import needs_avatars, needs_mhr
from mtcore.convert import MAP, REST_ALIGNED, VRM_CHILD, MhrToVrm, qinv, qmul, vrm_fk
from mtcore.mhr import FLEX_T, skeleton_state
from mtcore.paths import AVATARS
from mtcore.vrm import load_vrm


@needs_mhr
@needs_avatars
def test_bone_directions_survive_conversion(mhr, rng):
    """Random MHR poses -> VRM local rotations -> VRM FK: every limb/finger bone points where its MHR
    bone does (torso and shoulders map rest-to-rest instead; see the tests below)."""
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
            if c not in P or jd is None or b in REST_ALIGNED:
                continue
            dv, dm = P[c] - P[b], st[jn.index(jd), :3] - st[jn.index(MAP[b][0]), :3]
            cos = np.dot(dv, dm) / np.linalg.norm(dv) / np.linalg.norm(dm)
            worst = max(worst, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    assert worst < 0.1


def _angle(a, b):
    """Rotation angle between quaternions, deg (the MHR state is float32: ~0.1 deg of rounding)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, -1)), 0, 1)))


@needs_mhr
@needs_avatars
def test_mhr_rest_is_avatar_rest_for_torso_and_shoulders(mhr):
    """MHR's upright rest gives the avatar's rest for hips..head and the shoulders (no copied S-curve)."""
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn)))
    _, _, vrm_rest = load_vrm(AVATARS / "AvatarSample_D.vrm")
    L = MhrToVrm(jn, rest, vrm_rest).locals(rest[None])
    for b in REST_ALIGNED:
        assert _angle(L[b][0], np.array([0, 0, 0, 1.0])) < 0.2, b


@needs_mhr
@needs_avatars
def test_torso_follows_mhr_rotation_from_rest_and_arms_are_unaffected(mhr, rng, monkeypatch):
    """Torso/shoulder bones turn in the world exactly as their MHR joints turn away from rest; the arm
    bones' world rotations (after FK) are the same as with direction matching everywhere."""
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn)))
    _, _, vrm_rest = load_vrm(AVATARS / "AvatarSample_D.vrm")
    conv = MhrToVrm(jn, rest, vrm_rest)
    pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
    mp = np.zeros((20, len(pn)), np.float32)
    mp[:, pose_idx] = rng.uniform(-0.6, 0.6, (20, len(pose_idx)))
    st = skeleton_state(ch, mp)
    _, G = vrm_fk(conv.locals(st), vrm_rest)
    for b in REST_ALIGNED:
        j = jn.index(MAP[b][0])
        assert _angle(G[b], qmul(st[:, j, 3:7], qinv(rest[j, 3:7]))).max() < 0.2, b
    glob = conv.globals(st)
    for s in ("left", "right"):
        for b in (f"{s}UpperArm", f"{s}LowerArm", f"{s}Hand"):
            assert _angle(G[b], glob[b]).max() < 0.2, b


def _twist_deg(q, axis):
    """Signed twist angle of q about a unit axis, deg."""
    from mtcore.convert import twist
    t = twist(np.asarray(q, float), axis)
    return float((np.degrees(2 * np.arctan2(t[:3] @ axis, t[3])) + 180) % 360 - 180)


@needs_mhr
@needs_avatars
def test_forearm_carries_the_roll_and_the_wrist_only_bends(mhr, rng):
    """MHR's wrist_twist rest frame differs from lowarm's by ~114 deg about the forearm: only the change since rest
    may count (it once put ~130 deg on lowerArm that the hand undid: a wrung forearm). A wrist can't roll, so the VRM
    hand keeps no twist about the forearm in any pose, and a pure 40 deg forearm twist (wrist_twist and everything
    below it) turns lowerArm by 40 deg."""
    from mtcore.convert import qrot
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn))).astype(np.float64)
    _, _, vrm_rest = load_vrm(AVATARS / "AvatarSample_D.vrm")
    conv = MhrToVrm(jn, rest, vrm_rest)
    parents = list(ch.skeleton.joint_parents)
    pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
    poses = [rest]
    for _ in range(10):
        mp = np.zeros(len(pn), np.float32)
        mp[pose_idx] = rng.uniform(-0.6, 0.6, len(pose_idx))
        poses.append(skeleton_state(ch, mp).astype(np.float64))
    for side, m in (("left", "l"), ("right", "r")):
        axis = vrm_rest[f"{side}Hand"] - vrm_rest[f"{side}LowerArm"]
        axis /= np.linalg.norm(axis)
        for st in poses:
            assert abs(_twist_deg(conv.locals(st)[f"{side}Hand"], axis)) < 0.5

        lo, wt, wr = (jn.index(f"{m}_{n}") for n in ("lowarm", "wrist_twist", "wrist"))
        d = rest[wr, :3] - rest[lo, :3]
        d /= np.linalg.norm(d)
        R = np.append(d * np.sin(np.radians(20)), np.cos(np.radians(20)))  # 40 deg about the forearm, world frame
        st = rest.copy()
        below = {wt}
        for j in range(len(jn)):  # parents come before children
            if parents[j] in below:
                below.add(j)
        for j in below:
            st[j, 3:7] = qmul(R, rest[j, 3:7])
            st[j, :3] = rest[wt, :3] + qrot(R, rest[j, :3] - rest[wt, :3])
        L0, L = conv.locals(rest), conv.locals(st)
        turn = _twist_deg(qmul(qinv(L0[f"{side}LowerArm"]), L[f"{side}LowerArm"]), axis)
        assert abs(abs(turn) - 40.0) < 1.0
        assert abs(_twist_deg(L[f"{side}Hand"], axis)) < 0.5
