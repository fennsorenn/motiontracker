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
