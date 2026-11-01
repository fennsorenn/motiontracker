import numpy as np

from conftest import needs_mhr
from mtcore.landmarks import MP33, LandmarkModel
from mtcore.mhr import FLEX_T, skeleton_state


@needs_mhr
def test_rest_landmarks_match_mesh_and_sides(mhr):
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn)))
    lm = LandmarkModel(ch, rest)
    L = lm(rest)
    V = np.asarray(ch.mesh.vertices)
    for name, v in lm.vertex.items():  # rigid carry reproduces the vertex at rest
        assert np.allclose(L[MP33.index(name)], V[v], atol=1e-3), name
    for k, name in enumerate(MP33):  # person's left = MHR +x
        if name.startswith("left_") or name.endswith("_left"):
            assert L[k, 0] > 0, name
            j = MP33.index(name.replace("left", "right"))
            assert np.allclose(L[k] * [-1, 1, 1], L[j], atol=0.5), name  # rest mesh is symmetric


@needs_mhr
def test_face_landmarks_follow_head(mhr, rng):
    ch, jn, pn = mhr
    rest = skeleton_state(ch, np.zeros(len(pn)))
    lm = LandmarkModel(ch, rest)
    pose_idx = [i for i, n in enumerate(pn) if not n.startswith(("scale_", "root_t")) and n not in FLEX_T]
    mp = np.zeros((10, len(pn)), np.float32)
    mp[:, pose_idx] = rng.uniform(-0.4, 0.4, (10, len(pose_idx)))
    st = skeleton_state(ch, mp)
    L = lm(st)
    head = st[:, jn.index("c_head"), :3]
    rest_d = np.linalg.norm(lm(rest)[:11] - rest[jn.index("c_head"), :3], axis=-1)
    d = np.linalg.norm(L[:, :11] - head[:, None], axis=-1)
    # face points keep their distance to the head joint (mouth corners move a little with the jaw)
    assert np.abs(d - rest_d).max() < 1.5
