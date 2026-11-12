import struct

import numpy as np

from mtservice.vmc import VmcSender, rest_offsets


def _str(buf, o):
    end = buf.index(b"\0", o)
    s = buf[o:end].decode()
    return s, (end + 4) & ~3


def parse(buf):
    """Minimal OSC reader (bundle of messages with s/f/i args), like vspark's parsePacket."""
    assert buf[:8] == b"#bundle\0"
    o, msgs = 16, []
    while o < len(buf):
        (n,) = struct.unpack(">i", buf[o:o + 4])
        m = buf[o + 4:o + 4 + n]
        o += 4 + n
        addr, p = _str(m, 0)
        tags, p = _str(m, p)
        args = []
        for t in tags[1:]:
            if t == "s":
                v, p = _str(m, p)
            elif t == "f":
                (v,), p = struct.unpack(">f", m[p:p + 4]), p + 4
            else:
                (v,), p = struct.unpack(">i", m[p:p + 4]), p + 4
            args.append(v)
        msgs.append((addr, args))
    return msgs


def test_bone_packet_roundtrips_through_vsparks_mapping():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    s = VmcSender()
    s.set_rest({"hips": np.array([0, 1.0, 0]), "spine": np.array([0, 1.1, 0.01])})
    msgs = parse(s.packet(1.5, {"spine": q, "unknownBone": q}, {"c_head": 0.5}, present=False))
    s.close()
    bones = [a for addr, a in msgs if addr == "/VMC/Ext/Bone/Pos"]
    assert len(bones) == 1 and bones[0][0] == "Spine"
    x, y, z, w = bones[0][4:8]
    np.testing.assert_allclose([x, -y, -z, w], q, atol=1e-6)  # vspark: (x, -y, -z, w)
    np.testing.assert_allclose(bones[0][1:4], [0, 0.1, 0.01], atol=1e-6)  # rest offset, x mirrored
    assert ("/MT/Present", [0]) in msgs
    assert any(addr == "/MT/Conf" and a[0] == "c_head" for addr, a in msgs)


def test_rest_offsets_skip_missing_parents():
    off = rest_offsets({"hips": np.array([0, 1.0, 0]), "chest": np.array([0, 1.3, 0])})  # no spine
    np.testing.assert_allclose(off["chest"], [0, 0.3, 0])
