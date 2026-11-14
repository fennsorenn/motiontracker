"""Finger bones from MediaPipe hand world landmarks: a line-for-line port of vspark's algorithmic finger
solve (`hand_landmarks_to_bones.ts`, frozen copy in `baseline/vendor/backend/signal/nodes/`), checked
against it in tests/test_hands.py. Stopgap until the learned hand model (PLAN.md §7.5).

Input: 21 hand world landmarks of one side, as the detector returns them (mirrored frame, so "left" is
the person's left). Output: VRM finger bones -> local quaternion (x, y, z, w) in vspark's pose space,
the same space as the lifter's bones. The hand (wrist) bone is not set: the body model drives it.
Plain floats with `math`, as the TypeScript, so the two agree to rounding.
"""

import math

WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 0, 1, 2, 3, 4
FINGERS = (  # (mcp, pip, dip, tip, bone name stem)
    (5, 6, 7, 8, "Index"),
    (9, 10, 11, 12, "Middle"),
    (13, 14, 15, 16, "Ring"),
    (17, 18, 19, 20, "Little"),
)
INDEX_MCP, MIDDLE_MCP, LITTLE_MCP = 5, 9, 17

FLEX_MIN, FLEX_MAX = -0.2, 1.9
SPREAD_MAX = 0.45
THUMB_SPLAY = 0.6
THUMB_CMC_MAX = 1.2
THUMB_FLEX_MIN, THUMB_FLEX_MAX = -0.35, 1.6
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def finger_bones(side):
    """The 15 VRM finger bones of one side ("left" / "right")."""
    out = []
    for *_, stem in FINGERS:
        out += [f"{side}{stem}Proximal", f"{side}{stem}Intermediate", f"{side}{stem}Distal"]
    return out + [f"{side}ThumbMetacarpal", f"{side}ThumbProximal", f"{side}ThumbDistal"]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _len(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _norm(v):
    n = _len(v)
    return (0.0, 0.0, 1.0) if n < 1e-9 else (v[0] / n, v[1] / n, v[2] / n)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rot_from_to(f, t):
    d = _dot(f, t)
    if d > 0.9999:
        return IDENTITY
    if d < -0.9999:
        perp = (1.0, 0.0, 0.0) if abs(f[0]) < 0.9 else (0.0, 1.0, 0.0)
        ax = _norm(_cross(f, perp))
        return (ax[0], ax[1], ax[2], 0.0)
    ax = _norm(_cross(f, t))
    s = math.sin(math.acos(d) / 2)
    return (ax[0] * s, ax[1] * s, ax[2] * s, math.cos(math.acos(d) / 2))


def _qvec(q, v):
    x, y, z, w = q
    ix = w * v[0] + y * v[2] - z * v[1]
    iy = w * v[1] + z * v[0] - x * v[2]
    iz = w * v[2] + x * v[1] - y * v[0]
    iw = -x * v[0] - y * v[1] - z * v[2]
    return (ix * w + iw * -x + iy * -z - iz * -y,
            iy * w + iw * -y + iz * -x - ix * -z,
            iz * w + iw * -z + ix * -y - iy * -x)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def _axis_angle(axis, angle):
    s = math.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


THUMB_CURL = _norm((0.0, -0.5, -1.0))


def _hand_to_world(pts, side):
    wrist = pts[WRIST]
    finger_axis = _norm(_sub(pts[MIDDLE_MCP], wrist))
    palm = _norm(_cross(_sub(pts[INDEX_MCP], wrist), _sub(pts[LITTLE_MCP], wrist)))
    if side == "right":
        palm = _scale(palm, -1)
    p = _scale(finger_axis, _dot(palm, finger_axis))
    palm = _norm((palm[0] - p[0], palm[1] - p[1], palm[2] - p[2]))
    vrm_finger_axis = (1.0, 0.0, 0.0) if side == "left" else (-1.0, 0.0, 0.0)
    swing = _rot_from_to(vrm_finger_axis, finger_axis)
    d = _qvec(swing, (0.0, 1.0, 0.0))
    pa = _dot(d, finger_axis)
    d = _norm((d[0] - finger_axis[0] * pa, d[1] - finger_axis[1] * pa, d[2] - finger_axis[2] * pa))
    pb = _dot(palm, finger_axis)
    n = _norm((palm[0] - finger_axis[0] * pb, palm[1] - finger_axis[1] * pb, palm[2] - finger_axis[2] * pb))
    twist = math.atan2(_dot(_cross(d, n), finger_axis), _dot(d, n))
    s = math.sin(twist / 2)
    twist_q = (finger_axis[0] * s, finger_axis[1] * s, finger_axis[2] * s, math.cos(twist / 2))
    return _qmul(twist_q, swing)


def _signed_bend(pts, a, b, c, frame_flex, hyper_min):
    u = _norm(_sub(pts[b], pts[a]))
    v = _norm(_sub(pts[c], pts[b]))
    bend = math.acos(_clamp(_dot(u, v), -1, 1))
    return -bend if frame_flex < 0 and bend <= -hyper_min else bend


def _finger_segment(pts, i_from, i_to, parent, rest, i_prev, allow_spread=False):
    local = _qvec(_qinv(parent), _norm(_sub(pts[i_to], pts[i_from])))
    palm = (0.0, -1.0, 0.0)
    flex = math.atan2(_dot(local, palm), _dot(local, rest))
    if not allow_spread:
        flex = _signed_bend(pts, i_prev, i_from, i_to, flex, FLEX_MIN)
    q_flex = _axis_angle(_norm(_cross(rest, palm)), _clamp(flex, FLEX_MIN, FLEX_MAX))
    if not allow_spread:
        return q_flex
    side_axis = _norm(_cross(palm, rest))
    planar = (local[0], 0.0, local[2])
    pl = _len(planar)
    if pl < 0.2:
        return q_flex
    planar = (planar[0] / pl, planar[1] / pl, planar[2] / pl)
    spread = math.atan2(_dot(planar, side_axis), _dot(planar, rest))
    return _qmul(_axis_angle((0.0, -1.0, 0.0), _clamp(spread, -SPREAD_MAX, SPREAD_MAX)), q_flex)


def _thumb_metacarpal(pts, hand, rest):
    local = _qvec(_qinv(hand), _norm(_sub(pts[THUMB_MCP], pts[THUMB_CMC])))
    angle = math.acos(_clamp(_dot(rest, local), -1, 1))
    if angle < 1e-6:
        return IDENTITY
    return _axis_angle(_norm(_cross(rest, local)), min(angle, THUMB_CMC_MAX))


def _thumb_hinge(pts, i_prev, i_from, i_to, parent, rest):
    hinge = _norm(_cross(rest, THUMB_CURL))
    curl = _cross(hinge, rest)
    local = _qvec(_qinv(parent), _norm(_sub(pts[i_to], pts[i_from])))
    frame_flex = math.atan2(_dot(local, curl), _dot(local, rest))
    flex = _signed_bend(pts, i_prev, i_from, i_to, frame_flex, THUMB_FLEX_MIN)
    return _axis_angle(hinge, _clamp(flex, THUMB_FLEX_MIN, THUMB_FLEX_MAX))


def solve_hand(points, side):
    """points: 21 hand world landmarks (rows of x, y, z); side: "left" / "right".
    Returns {VRM finger bone: (x, y, z, w)}; empty for fewer than 21 points."""
    if len(points) < 21:
        return {}
    pts = [tuple(float(c) for c in p[:3]) for p in points]
    hand = _hand_to_world(pts, side)
    rest = (1.0, 0.0, 0.0) if side == "left" else (-1.0, 0.0, 0.0)
    out = {}
    for mcp, pip, dip, tip, stem in FINGERS:
        q_mcp = _finger_segment(pts, mcp, pip, hand, rest, WRIST, True)
        w_mcp = _qmul(hand, q_mcp)
        q_pip = _finger_segment(pts, pip, dip, w_mcp, rest, mcp)
        w_pip = _qmul(w_mcp, q_pip)
        q_dip = _finger_segment(pts, dip, tip, w_pip, rest, pip)
        out[f"{side}{stem}Proximal"], out[f"{side}{stem}Intermediate"], out[f"{side}{stem}Distal"] = q_mcp, q_pip, q_dip
    thumb_rest = _norm((rest[0] * math.cos(THUMB_SPLAY), 0.0, math.sin(THUMB_SPLAY)))
    q_cmc = _thumb_metacarpal(pts, hand, thumb_rest)
    w_cmc = _qmul(hand, q_cmc)
    q_mcp = _thumb_hinge(pts, THUMB_CMC, THUMB_MCP, THUMB_IP, w_cmc, thumb_rest)
    w_mcp = _qmul(w_cmc, q_mcp)
    q_ip = _thumb_hinge(pts, THUMB_MCP, THUMB_IP, THUMB_TIP, w_mcp, thumb_rest)
    out[f"{side}ThumbMetacarpal"], out[f"{side}ThumbProximal"], out[f"{side}ThumbDistal"] = q_cmc, q_mcp, q_ip
    return out


def solve_hands(obs):
    """Detector output -> finger bones of every detected hand (world landmarks)."""
    out = {}
    for side in ("left", "right"):
        pts = obs.get(f"{side}Hand")
        if pts is not None:
            out.update(solve_hand(pts, side))
    return out
