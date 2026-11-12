"""VMC (OSC over UDP) output.

Bone rotations go out as /VMC/Ext/Bone/Pos with Unity HumanBodyBones names. Coordinates: vspark maps
a VMC rotation (x, y, z, w) to its pose space as (x, -y, -z, w) (rhylive_bone_mapper), the space the
lifter's VRM output is already in, so the sender applies the same (self-inverse) map. Positions are
the avatar's rest offsets from the parent bone, mapped the same way (x, y, z) -> (-x, y, z).
Face blendshapes go out as /VMC/Ext/Blend/Val <ARKit name> <0..1> per shape, then one /VMC/Ext/Blend/Apply
(the VMC "perfect sync" naming; the person's own left/right).
Extensions (receivers ignore unknown addresses): /MT/Present, /MT/Conf <joint> <0..1>. The namespace
name is provisional (PLAN.md §7.2, [OPEN]).
"""

import socket
import struct

UNITY_NAMES = {
    "hips": "Hips", "spine": "Spine", "chest": "Chest", "upperChest": "UpperChest", "neck": "Neck", "head": "Head",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftUpperArm", "leftLowerArm": "LeftLowerArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightUpperArm", "rightLowerArm": "RightLowerArm", "rightHand": "RightHand",
}
VRM_PARENT = {"spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
              "leftShoulder": "upperChest", "rightShoulder": "upperChest", "leftUpperArm": "leftShoulder",
              "rightUpperArm": "rightShoulder", "leftLowerArm": "leftUpperArm", "rightLowerArm": "rightUpperArm",
              "leftHand": "leftLowerArm", "rightHand": "rightLowerArm"}


def _pad(b):
    return b + b"\0" * (4 - len(b) % 4)


def message(address, *args):
    """One OSC message; args are str, float or int."""
    tags, data = ",", b""
    for a in args:
        if isinstance(a, str):
            tags += "s"
            data += _pad(a.encode())
        elif isinstance(a, int) and not isinstance(a, bool):
            tags += "i"
            data += struct.pack(">i", a)
        else:
            tags += "f"
            data += struct.pack(">f", float(a))
    return _pad(address.encode()) + _pad(tags.encode()) + data


def bundle(messages, timetag=1):
    """OSC bundle (timetag 1 = immediately)."""
    out = _pad(b"#bundle") + struct.pack(">Q", timetag)
    for m in messages:
        out += struct.pack(">i", len(m)) + m
    return out


def rest_offsets(rest_positions):
    """VRM bone -> local rest offset from its parent (normalized VRM), for the Bone/Pos position fields."""
    out = {}
    for b, p in rest_positions.items():
        parent = VRM_PARENT.get(b)
        while parent is not None and parent not in rest_positions:
            parent = VRM_PARENT.get(parent)
        out[b] = [float(v) for v in (p if parent is None else p - rest_positions[parent])]
    return out


class VmcSender:
    def __init__(self, host="127.0.0.1", port=39539):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.offsets = {}

    def set_rest(self, rest_positions):
        self.offsets = rest_offsets(rest_positions)

    def packet(self, t, bones, conf=None, present=True, blendshapes=None):
        msgs = [message("/VMC/Ext/OK", 1), message("/VMC/Ext/T", float(t)),
                message("/VMC/Ext/Root/Pos", "root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
        for b, q in bones.items():
            name = UNITY_NAMES.get(b)
            if name is None:
                continue
            px, py, pz = self.offsets.get(b, (0.0, 0.0, 0.0))
            msgs.append(message("/VMC/Ext/Bone/Pos", name, -px, py, pz, q[0], -q[1], -q[2], q[3]))
        if blendshapes:
            msgs += [message("/VMC/Ext/Blend/Val", name, v) for name, v in blendshapes.items()]
            msgs.append(message("/VMC/Ext/Blend/Apply"))
        msgs.append(message("/MT/Present", 1 if present else 0))
        for j, c in (conf or {}).items():
            msgs.append(message("/MT/Conf", j, c))
        return bundle(msgs)

    def send(self, t, bones, conf=None, present=True, blendshapes=None):
        self.sock.sendto(self.packet(t, bones, conf, present, blendshapes), self.addr)

    def close(self):
        self.sock.close()
