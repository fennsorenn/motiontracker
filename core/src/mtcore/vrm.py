"""Minimal VRM (0.x and 1.0) reader: humanoid bone -> rest world position (metres)."""

import json
import struct

import numpy as np

VRM0_TO_1 = {"leftThumbProximal": "leftThumbMetacarpal", "leftThumbIntermediate": "leftThumbProximal",
             "rightThumbProximal": "rightThumbMetacarpal", "rightThumbIntermediate": "rightThumbProximal"}


def _gltf_json(path):
    with open(path, "rb") as f:
        f.read(12)
        clen, _ = struct.unpack("<I4s", f.read(8))
        return json.loads(f.read(clen))


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_vrm(path):
    """Returns (version, meta dict, {vrm1_bone_name: world position})."""
    g = _gltf_json(path)
    ext = g.get("extensions", {})
    if "VRMC_vrm" in ext:
        v = ext["VRMC_vrm"]
        bones = {k: b["node"] for k, b in v["humanoid"]["humanBones"].items()}
        version = "1.0"
    else:
        v = ext["VRM"]
        bones = {VRM0_TO_1.get(b["bone"], b["bone"]): b["node"] for b in v["humanoid"]["humanBones"]}
        version = "0.x"
    nodes = g["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    cache = {}

    def M(i):
        if i not in cache:
            n = nodes[i]
            m = np.eye(4)
            if "matrix" in n:
                m = np.array(n["matrix"]).reshape(4, 4).T
            else:
                m[:3, :3] = _quat_mat(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]))
                m[:3, 3] = n.get("translation", [0, 0, 0])
            cache[i] = M(parent[i]) @ m if i in parent else m
        return cache[i]

    pos = {k: M(i)[:3, 3].copy() for k, i in bones.items()}
    if version == "0.x":  # VRM 0.x faces -Z; VRM 1.0 (and MHR) face +Z: turn 180 degrees about Y
        pos = {k: p * np.array([-1.0, 1.0, -1.0]) for k, p in pos.items()}
    return version, v.get("meta", {}), pos


_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _glb(path):
    with open(path, "rb") as f:
        f.read(12)
        clen, _ = struct.unpack("<I4s", f.read(8))
        g = json.loads(f.read(clen))
        blen, _ = struct.unpack("<I4s", f.read(8))
        return g, f.read(blen)


def _accessor(g, binary, i):
    a = g["accessors"][i]
    v = g["bufferViews"][a["bufferView"]]
    dt, n = np.dtype(_COMPONENT[a["componentType"]]), _NCOMP[a["type"]]
    start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", dt.itemsize * n)
    raw = np.frombuffer(binary, np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * n), (stride, 1))
    out = np.ascontiguousarray(rows).view(dt).reshape(a["count"], n)
    if a.get("normalized"):
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


def load_vrm_mesh(path, skip=("Hair",)):
    """Skinned mesh of a VRM at rest: (vertices (V, 3) world metres, humanoid bone per vertex (V,)).

    Each vertex goes to the humanoid bone of its strongest joint, or that joint's nearest humanoid ancestor (bust,
    sleeve and skirt secondaries count for the body part they hang from); vertices whose strongest joint's name
    contains one of `skip`, and meshes whose name does (the hair: the human body it is compared with has none), are left out. Same facing as `load_vrm` (VRM 0.x turned to +Z)."""
    g, binary = _glb(path)
    ext = g.get("extensions", {})
    if "VRMC_vrm" in ext:
        bones = {b["node"]: k for k, b in ext["VRMC_vrm"]["humanoid"]["humanBones"].items()}
        v0 = False
    else:
        bones = {b["node"]: VRM0_TO_1.get(b["bone"], b["bone"]) for b in ext["VRM"]["humanoid"]["humanBones"]}
        v0 = True
    nodes = g["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    cache = {}

    def M(i):
        if i not in cache:
            n = nodes[i]
            m = np.eye(4)
            if "matrix" in n:
                m = np.array(n["matrix"]).reshape(4, 4).T
            else:
                m[:3, :3] = _quat_mat(n.get("rotation", [0, 0, 0, 1])) * np.array(n.get("scale", [1, 1, 1]))
                m[:3, 3] = n.get("translation", [0, 0, 0])
            cache[i] = M(parent[i]) @ m if i in parent else m
        return cache[i]

    def human(i):
        while i is not None and i not in bones:
            i = parent.get(i)
        return bones.get(i)

    verts, labels = [], []
    for ni, n in enumerate(nodes):
        if "mesh" not in n or "skin" not in n or any(s in n.get("name", "") for s in skip):
            continue
        skin = g["skins"][n["skin"]]
        joints = skin["joints"]
        ibm = _accessor(g, binary, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        J = np.stack([M(j) @ ibm[k] for k, j in enumerate(joints)])  # (nj, 4, 4) rest skinning matrices
        for p in g["meshes"][n["mesh"]]["primitives"]:
            at = p["attributes"]
            if "JOINTS_0" not in at:
                continue
            pos = _accessor(g, binary, at["POSITION"]).astype(np.float64)
            jt = _accessor(g, binary, at["JOINTS_0"]).astype(int)
            wt = _accessor(g, binary, at["WEIGHTS_0"]).astype(np.float64)
            ph = np.concatenate([pos, np.ones((len(pos), 1))], 1)
            world = np.einsum("vk,vkij,vj->vi", wt, J[jt], ph)[:, :3]
            strongest = jt[np.arange(len(jt)), wt.argmax(1)]
            for k in range(len(pos)):
                j = joints[strongest[k]]
                name = nodes[j].get("name", "")
                if any(s in name for s in skip):
                    continue
                b = human(j)
                if b is not None:
                    verts.append(world[k])
                    labels.append(b)
    verts = np.array(verts)
    if v0:
        verts = verts * np.array([-1.0, 1.0, -1.0])
    return verts, np.array(labels)
