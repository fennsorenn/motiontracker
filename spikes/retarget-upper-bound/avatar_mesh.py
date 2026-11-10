"""VRoid VRM avatars: rest-pose skin vertices grouped by body part, for contact measurements.

Vertices are skinned to their rest (bind) positions in the avatar's world frame (cm, facing +Z like
VRM 1.0 and MHR; VRM 0.x is turned 180 degrees about Y as in `mtcore.vrm.load_vrm`). Each vertex is
assigned to the humanoid bone of its dominant skin joint (walking up the node tree until a humanoid
bone is found), so a part can be posed rigidly with that bone's global rotation:
    v = P[bone] + G[bone] (v_rest - rest[bone])     (normalized bones: identity rest rotations)
Groups: 'head' = the Face mesh (VRoid: face skin, ears, eyes, brows; no hair), 'leftHand' /
'rightHand' = Body-mesh vertices on the hand or its fingers.
"""

import json
import struct

import numpy as np

from mtcore.vrm import VRM0_TO_1, _quat_mat

COMP = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + n])
    off = 20 + n
    bn = struct.unpack("<I", b[off:off + 4])[0]
    return g, b[off + 8:off + 8 + bn]


def _accessor(g, binary, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    dt, k = COMP[a["componentType"]], NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    item = np.dtype(dt).itemsize * k
    if stride and stride != item:
        raw = np.frombuffer(binary, np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)[:, :item]
        out = np.ascontiguousarray(raw).view(dt).reshape(a["count"], k)
    else:
        out = np.frombuffer(binary, dt, count=a["count"] * k, offset=start).reshape(a["count"], k)
    return out.astype(np.float64) if dt == np.float32 else out.astype(np.int64)


def load_avatar(path):
    """-> dict(rest: {bone: xyz cm}, groups: {group: (verts (N,3) cm, bone names (N,), normals (N,3))})."""
    g, binary = _glb(path)
    ext = g.get("extensions", {})
    if "VRMC_vrm" in ext:
        hb = {k: b["node"] for k, b in ext["VRMC_vrm"]["humanoid"]["humanBones"].items()}
        flip = np.ones(3)
    else:
        hb = {VRM0_TO_1.get(b["bone"], b["bone"]): b["node"] for b in ext["VRM"]["humanoid"]["humanBones"]}
        flip = np.array([-1.0, 1.0, -1.0])
    node_bone = {n: b for b, n in hb.items()}
    nodes = g["nodes"]
    parent = {c: i for i, nd in enumerate(nodes) for c in nd.get("children", [])}
    cache = {}

    def M(i):
        if i not in cache:
            nd = nodes[i]
            m = np.eye(4)
            if "matrix" in nd:
                m = np.array(nd["matrix"]).reshape(4, 4).T
            else:
                m[:3, :3] = _quat_mat(nd.get("rotation", [0, 0, 0, 1])) * np.array(nd.get("scale", [1, 1, 1]))
                m[:3, 3] = nd.get("translation", [0, 0, 0])
            cache[i] = M(parent[i]) @ m if i in parent else m
        return cache[i]

    def bone_of(i):
        while i is not None and i not in node_bone:
            i = parent.get(i)
        return node_bone.get(i)

    rest = {b: M(n)[:3, 3] * flip * 100.0 for b, n in hb.items()}
    hand_bones = {s: {b for b in hb if b.startswith(s) and (b == f"{s}Hand" or any(f in b for f in ("Thumb", "Index", "Middle", "Ring", "Little")))}
                  for s in ("left", "right")}
    groups = {"head": ([], [], []), "leftHand": ([], [], []), "rightHand": ([], [], [])}
    for i, nd in enumerate(nodes):
        if "mesh" not in nd or "skin" not in nd:
            continue
        mesh_name = (nd.get("name") or "") + (g["meshes"][nd["mesh"]].get("name") or "")
        if "hair" in mesh_name.lower():
            continue
        skin = g["skins"][nd["skin"]]
        joints = skin["joints"]
        ibm = _accessor(g, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        J = np.stack([M(j) @ ibm[k] for k, j in enumerate(joints)])  # joint skinning matrices at rest
        jbone = [bone_of(j) for j in joints]
        for prim in g["meshes"][nd["mesh"]]["primitives"]:
            at = prim["attributes"]
            if "JOINTS_0" not in at:
                continue
            V = _accessor(g, binary, at["POSITION"])
            Jx, Wx = _accessor(g, binary, at["JOINTS_0"]), _accessor(g, binary, at["WEIGHTS_0"])
            Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
            Vr = np.einsum("nk,nkij,nj->ni", Wx, J[Jx], Vh)[:, :3] * flip * 100.0
            Nrm = _accessor(g, binary, at["NORMAL"])
            Nr = np.einsum("nk,nkij,nj->ni", Wx, J[Jx][..., :3, :3], Nrm) * flip
            Nr /= np.linalg.norm(Nr, axis=1, keepdims=True) + 1e-12
            dom = Jx[np.arange(len(V)), Wx.argmax(1)]
            bones = np.array([jbone[d] for d in dom], dtype=object)
            if "face" in mesh_name.lower():
                groups["head"][0].append(Vr)
                groups["head"][1].append(np.where(bones == None, "head", bones))  # noqa: E711
                groups["head"][2].append(Nr)
            else:
                for s in ("left", "right"):
                    m = np.isin(bones, list(hand_bones[s]))
                    groups[f"{s}Hand"][0].append(Vr[m])
                    groups[f"{s}Hand"][1].append(bones[m])
                    groups[f"{s}Hand"][2].append(Nr[m])
    groups = {k: (np.concatenate(v[0]), np.concatenate(v[1]).astype(str), np.concatenate(v[2])) for k, v in groups.items() if v[0]}
    return {"rest": rest, "groups": groups}
