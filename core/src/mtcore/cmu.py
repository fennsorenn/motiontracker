"""CMU mocap ASF/AMC reader with forward kinematics -> joint end positions in centimetres.

Convention as in common CMU parsers: bone matrix = parent * C * R(motion) * C^-1, with
C = R(axis) and Euler angles applied as R = Rz * Ry * Rx; bone end = parent end + length *
matrix @ direction. Length units: ASF 'length 0.45' means 1 unit = 1/0.45 inch.
"""

from pathlib import Path

import numpy as np

from .paths import CMU_ROOT


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_xyz(rx, ry, rz):
    return _rot(2, rz) @ _rot(1, ry) @ _rot(0, rx)


class Skeleton:
    def __init__(self, asf_path):
        text = Path(asf_path).read_text().splitlines()
        self.unit_cm = 2.54
        self.bones = {}  # name -> dict(direction, length, C, dof)
        self.children = {}
        self.root_order, self.root_axis = None, (0, 0, 0)
        section, cur = None, None
        for raw in text:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(":"):
                section = line.split()[0][1:]
                if section == "units":
                    pass
                continue
            parts = line.split()
            if section == "units" and parts[0] == "length":
                self.unit_cm = 2.54 / float(parts[1])
            elif section == "root":
                if parts[0] == "order":
                    self.root_order = parts[1:]
                elif parts[0] == "orientation":
                    self.root_axis = tuple(float(x) for x in parts[1:4])
            elif section == "bonedata":
                if parts[0] == "begin":
                    cur = {"dof": []}
                elif parts[0] == "end":
                    self.bones[cur["name"]] = cur
                elif parts[0] == "name":
                    cur["name"] = parts[1]
                elif parts[0] == "direction":
                    cur["direction"] = np.array([float(x) for x in parts[1:4]])
                elif parts[0] == "length":
                    cur["length"] = float(parts[1])
                elif parts[0] == "axis":
                    cur["C"] = _euler_xyz(*[float(x) for x in parts[1:4]])
                elif parts[0] == "dof":
                    cur["dof"] = parts[1:]
            elif section == "hierarchy":
                if parts[0] in ("begin", "end"):
                    continue
                self.children[parts[0]] = parts[1:]
        self.root_C = _euler_xyz(*self.root_axis)

    def fk(self, frame):
        """frame: {bone: [values]} -> {bone: end position (cm)}, plus 'root'."""
        r = frame["root"]
        vals = dict(zip(self.root_order, r))
        pos0 = np.array([vals.get("TX", 0), vals.get("TY", 0), vals.get("TZ", 0)])
        R0 = self.root_C @ _euler_xyz(vals.get("RX", 0), vals.get("RY", 0), vals.get("RZ", 0)) @ self.root_C.T
        P, M = {"root": pos0 * self.unit_cm}, {"root": R0}
        stack = [("root", c) for c in self.children.get("root", [])]
        while stack:
            parent, name = stack.pop()
            b = self.bones[name]
            ang = [0.0, 0.0, 0.0]
            for d, v in zip(b["dof"], frame.get(name, [])):
                ang["xyz".index(d[1])] = v
            M[name] = M[parent] @ b["C"] @ _euler_xyz(*ang) @ b["C"].T
            P[name] = P[parent] + b["length"] * self.unit_cm * (M[name] @ b["direction"])
            stack += [(name, c) for c in self.children.get(name, [])]
        return P


def read_amc(path):
    frames, cur = [], None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        parts = line.split()
        if parts[0].isdigit() and len(parts) == 1:
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[parts[0]] = [float(x) for x in parts[1:]]
    return frames


def clip_positions(subject, clip, step=4):
    """Joint end positions for every `step`-th frame (CMU is 120 fps; step 4 -> 30 fps)."""
    sid = f"{int(subject):02d}"
    d = CMU_ROOT / sid
    sk = Skeleton(d / f"{sid}.asf")
    frames = read_amc(d / f"{sid}_{int(clip):02d}.amc")[::step]
    names = list(sk.fk(frames[0]).keys())
    out = np.array([[sk.fk(f)[n] for n in names] for f in frames])
    return names, out
