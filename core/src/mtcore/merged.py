"""The merged pose, live: one detector frame in, VRM bones (body, legs, fingers), hip position and up direction out.

The same chain as the offline runner (`mttrain.merger.run.infer`), one frame at a time, in numpy plus onnxruntime
(three models from a merger bundle, `mttrain.merger.export`):
1. body model (`m1-v12+`, ONNX): rotations of the 23 output joints, their log scales, hip position (camera frame, m),
   its log scales and the up direction; streaming input features as `mtcore.features.StreamFeatures`;
2. pair hand model (`p3`, ONNX): per hand fingers, orientation, image position and size over the last HISTORY
   frames (10 FPS steps, `TRAIN_DT`), with the detector's confidence for each slot;
3. head pose from the face mesh (`mtcore.facepose`), rotated into MHR's head frame by the calibrated offset;
4. the base pose: the hand-written combination of the three (`mttrain.merger.batch.combine`);
5. the merger (ONNX) over its frame offsets (10 FPS steps): residuals on the base pose's global rotations, finger
   residuals, hip position and up residuals.
The hand model and the merger were trained on 10 FPS sequences and look back a fixed number of steps; live, each step
is picked by age (`TRAIN_DT` apart, the latest frame at or before it), as the body model's features already are, so
the runtime can process more frames per second (or drop some) without retraining.
`history_shift` (experiment, for 20 FPS): at 20 FPS neighbouring outputs share no history frames (each reads every
other frame), and the output alternates slightly from frame to frame. Shifting steps half a step older (50 ms) makes
neighbouring outputs mix both sets of frames: "every" shifts every step but the newest, "alternate" every second step
(by index), in all three histories. Measured on renders at 20 FPS (2026-11-23): both lower the alternation and the
visible shake a little, at ~0.15 deg more error.
onnxruntime is imported on use: `mtcore` itself stays numpy-only.
"""

import json
from pathlib import Path

import numpy as np

from .facepose import CanonicalFace, fit
from .features import OFFSETS, StreamFeatures
from .hand_features import canonical, palm_frame
from .lifter_pose import LifterPose

HAND_ASPECT = 4 / 3  # frame width / height (the recorder's 320x240)
POSE_HAND = {"l": [15, 17, 19, 21], "r": [16, 18, 20, 22]}  # Holistic pose points of each hand
MIRROR = np.diag([-1.0, 1.0, 1.0])
TRAIN_DT = 0.1  # history step of the hand model and the merger: they were trained on 10 FPS sequences
HISTORY_TOL = 0.02  # a frame up to this much younger than a step's age still counts (as features.StreamFeatures)
HISTORY_SHIFTS = ("none", "every", "alternate")
SHIFT = TRAIN_DT / 2


def shifted_ages(ages, mode):
    """History step ages (s) with `history_shift` applied: "every" step but the newest, or every "alternate" one
    (odd index), half a step older."""
    if mode not in HISTORY_SHIFTS:
        raise ValueError(f"history shift must be one of {HISTORY_SHIFTS}, not {mode!r}")
    return [a + SHIFT if (mode == "every" and i >= 1) or (mode == "alternate" and i % 2 == 1) else a
            for i, a in enumerate(ages)]


def pick_history(times, t, ages):
    """times: frame times, newest first; per age (s) the index of the latest frame at or before t - age (within
    HISTORY_TOL), or None when every kept frame is younger."""
    out = []
    for age in ages:
        due = t - age + HISTORY_TOL
        out.append(next((i for i, ft in enumerate(times) if ft <= due), None))
    return out


def prune_history(items, times, t, oldest_age):
    """Drop frames older than the newest one the oldest age (plus a shift, so it can change live) still needs
    (both lists newest first)."""
    due = t - oldest_age - SHIFT + HISTORY_TOL
    keep = next((i for i, ft in enumerate(times) if ft <= due), None)
    if keep is not None:
        del items[keep + 1:], times[keep + 1:]


# --- rotations (numpy ports of mttrain.merger.batch) ------------------------------------------------------------
def to6d(R):
    """(..., 3, 3) -> (..., 6): the first two columns, column after column (`mttrain.merger.model.to6d`)."""
    return np.swapaxes(R[..., :, :2], -1, -2).reshape(R.shape[:-2] + (6,))


def geodesic(Ra, Rb):
    c = ((Ra * Rb).sum((-1, -2)) - 1) / 2
    return np.arccos(np.clip(c, -1 + 1e-6, 1 - 1e-6))


def slerp_mat(A, B, w):
    """Rotation from A (w = 0) to B (w = 1)."""
    D = A.T @ B
    ang = geodesic(D, np.eye(3))
    axis = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    th = w * ang
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return A @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K))


def globals_from_locals(L, parent_idx):
    G = np.empty_like(L)
    for k, p in enumerate(parent_idx[:len(L)]):
        G[k] = L[k] if p < 0 else G[p] @ L[k]
    return G


def locals_from_globals(G, parent_idx):
    L = np.empty_like(G)
    for k, p in enumerate(parent_idx[:len(G)]):
        L[k] = G[k] if p < 0 else G[p].T @ G[k]
    return L


def unmirror_frame(F):
    """Palm frame rows of mirrored points (right hands in the scene convention) -> frame of the unmirrored points."""
    G = F @ MIRROR
    return np.concatenate([G[:2], -G[2:]], 0)


# --- hand model input (numpy port of mttrain.hands.model2.frame_features and evaluate2.LearnedPair) -------------
def hand_image_points(L):
    """Image hand landmarks (21, >=3) -> x right, y down, z away in frame widths (`hands.imagecal.image_points`, 'l')."""
    L = np.asarray(L, dtype=np.float64)
    return np.stack([L[:, 0] - 0.5, (L[:, 1] - 0.5) / HAND_ASPECT, L[:, 2]], -1)


def hand_score(obs, side):
    """The detector's confidence for a hand slot (`hands.scorecal.hand_score`): Holistic gives none, so the mean
    visibility of the pose's hand points."""
    S = "left" if side == "l" else "right"
    if f"{S}HandScore" in obs:
        return float(obs[f"{S}HandScore"])
    pose = obs.get("pose2d")
    if pose is None:
        pose = obs.get("pose")
    if pose is not None and len(pose) and len(pose[0]) > 3:
        return float(np.mean([pose[k][3] for k in POSE_HAND[side]]))
    return 0.0


def hand_frame_features(img, canon, lost, mirror, score):
    """One hand, one frame -> (FRAME_DIM,) as `hands.model2.frame_features`."""
    if mirror:
        img = img * np.array([-1.0, 1.0, 1.0])
    _, R, size = palm_frame(img)
    f = np.concatenate([canon.reshape(-1), R[:2].reshape(-1), img[0, :2], [np.log(max(size, 1e-4))]])
    if lost:
        f = np.zeros_like(f)
    return np.concatenate([f, [float(lost), score]])


class LiveMerged:
    def __init__(self, bundle, head3d=True, threads=1):
        import onnxruntime as ort
        bundle = Path(bundle)
        self.meta = m = json.loads((bundle / "merger.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = lambda f: ort.InferenceSession(str(bundle / f), so, providers=["CPUExecutionProvider"])
        self.body_net, self.hand_net, self.merger_net = sess("body.onnx"), sess("hands.onnx"), sess("merger.onnx")
        sk = m["skeleton"]
        self.joints = sk["out_joints"]
        self.parent_idx = sk["out_parent_idx"]
        self.J = {n: i for i, n in enumerate(self.joints)}
        self.bones = m["output_bones"]
        self.default_rest = {b: np.asarray(p) for b, p in m["default_avatar"]["rest_positions"].items()}
        self.rest = self.default_rest
        self.pose = LifterPose.from_table(sk, self.rest, bones=self.bones)
        h = m["hands"]
        self.hand_history = h["history"]
        self.F_rest = {s: np.asarray(h["F_rest"][s]) for s in "lr"}
        self.out_idx = h["out_idx"]
        self.lo, self.hi = np.asarray(h["lo"]), np.asarray(h["hi"])
        c = m["combine"]
        self.conf_deg, self.max_bend, self.head_w = c["conf_deg"], c["max_bend_deg"], c["head_weight"]
        f = m["face"]
        self.canon = CanonicalFace.from_arrays(f["points"], f["ids"], f["weights"])
        self.face_offset, self.rest_head = np.asarray(f["offset"]), np.asarray(f["rest_head"])
        self.blend_names = m["blendshapes"]
        self.offsets = m["merger"]["offsets"]
        self.nj = m["merger"]["joints"]
        self.head3d = head3d
        # live retargeting (`relretarget.LiveRetarget`) to an avatar given as a VRM file; the human side is MHR's mean
        # body from `retarget.npz` (`mttrain.retarget_asset`). Built for every VRM avatar (when the bundle has the
        # mesh) so it can be switched on and off live; used only while `retarget_on` (default off: on tracked input it
        # helps separated hands but carries the tracker's gap into touching ones, PLAN.md §3.10a, 2026-11-16)
        self.skeleton = sk
        f = bundle / "retarget.npz"
        self.human_mesh = None
        if f.exists():
            z = np.load(f)
            self.human_mesh = (z["verts"].astype(np.float64), z["labels"])
        self.retarget_on = False
        self._history_shift = "none"
        self.retarget, self.retarget_error = None, None
        self.reset()

    # --- avatar -------------------------------------------------------------------------------------------------
    def set_avatar(self, rest_positions=None):
        """Rest positions only (None: the default avatar): no retargeting (it needs the avatar's mesh)."""
        self.rest = self.default_rest if rest_positions is None else {b: np.asarray(p, dtype=float) for b, p in rest_positions.items()}
        self.pose.set_avatar(self.rest)
        self.retarget, self.retarget_error = None, None

    def set_avatar_vrm(self, path):
        """Avatar from its VRM file: its rest pose, and the retargeting to it (used while `retarget_on`). A
        retargeting failure leaves the rotation copy and is kept in `retarget_error`."""
        from .vrm import load_vrm, load_vrm_mesh
        self.set_avatar(load_vrm(path)[2])
        if self.human_mesh is None:
            self.retarget_error = "the model bundle has no retarget.npz"
            return
        try:
            from .relretarget import ANCHORS, LiveRetarget
            verts, labels = load_vrm_mesh(path)
            anchors = [a for a in ANCHORS if a in self.bones]  # no leg bones sent: the receiver's legs are unknown
            self.retarget = LiveRetarget(self.skeleton, *self.human_mesh, self.rest, verts, labels, self.pose.conv,
                                         anchors=anchors)
        except Exception as e:
            self.retarget_error = f"{type(e).__name__}: {e}"

    @property
    def retarget_active(self):
        """Retargeting is applied: switched on and set up for the current avatar."""
        return self.retarget_on and self.retarget is not None

    @property
    def history_shift(self):
        return self._history_shift

    @history_shift.setter
    def history_shift(self, mode):
        """Switchable live: the histories keep half a step more than they need."""
        shifted_ages([], mode)  # validates
        self._history_shift = mode
        self.features.offsets = np.array(shifted_ages(list(OFFSETS), mode))

    def reset(self):
        self.features = StreamFeatures(self.head3d, shifted_ages(list(OFFSETS), self.history_shift))
        self.hands = []  # newest first: per frame the hand model's features {(slot, mirrored): (FRAME_DIM,)}
        self.hand_times = []  # their times (s), newest first
        self.frames = []  # merger frame inputs, newest first
        self.frame_times = []

    # --- parts --------------------------------------------------------------------------------------------------
    def _body(self, t, obs):
        present = "pose" in obs and "pose2d" in obs
        if present:
            world, vis, image = obs["pose"][:, :3], obs["pose"][:, 3], obs["pose2d"][:, :3]
        else:
            world, vis, image = np.zeros((33, 3)), np.zeros(33), np.zeros((33, 3))
        x = self.features.push(t, world, image, vis, present)
        rot, ls, root, root_ls, up = self.body_net.run(None, {"x": x[None].astype(np.float32)})
        log_s = -np.log(np.clip(np.exp(-ls[0].astype(np.float64)), 1e-6, None))  # as the offline runner (conf -> log)
        return present, rot[0].astype(np.float64), log_s, root[0].astype(np.float64), root_ls[0].astype(np.float64), up[0].astype(np.float64)

    def _hands(self, t, obs):
        img, canon, lost = np.zeros((2, 21, 3)), np.zeros((2, 21, 3)), np.ones(2, bool)
        score = np.array([hand_score(obs, s) for s in "lr"])
        for k, (side, S) in enumerate((("l", "left"), ("r", "right"))):
            if obs.get(f"{S}Hand2d") is not None and obs.get(f"{S}Hand") is not None:
                img[k] = hand_image_points(obs[f"{S}Hand2d"])
                canon[k] = canonical(np.asarray(obs[f"{S}Hand"])[:, :3], side)
                lost[k] = False
        self.hands.insert(0, {(k, m): hand_frame_features(img[k], canon[k], lost[k], m, score[k])
                              for k in range(2) for m in (False, True)})
        self.hand_times.insert(0, t)
        ages = shifted_ages([k * TRAIN_DT for k in range(self.hand_history)], self.history_shift)
        prune_history(self.hands, self.hand_times, t, (self.hand_history - 1) * TRAIN_DT)
        pad = {(k, m): hand_frame_features(np.zeros((21, 3)), np.zeros((21, 3)), True, m, 0.0)
               for k in range(2) for m in (False, True)}  # before the session's first frames: lost, score 0
        H = [pad if i is None else self.hands[i] for i in pick_history(self.hand_times, t, ages)]
        ff = lambda k, mirror: np.stack([h[(k, mirror)] for h in H])
        own = np.stack([ff(0, False), ff(1, True)]).astype(np.float32)
        other = np.stack([ff(1, False), ff(0, True)]).astype(np.float32)
        names = [o.name for o in self.hand_net.get_outputs()]
        out = dict(zip(names, (v.astype(np.float64) for v in self.hand_net.run(None, {"own": own, "other": other}))))
        hand = {}
        for k, s in enumerate("lr"):
            F = out["rot"][k] if s == "l" else unmirror_frame(out["rot"][k])
            hand[s] = {"G": F.T @ self.F_rest[s], "rot_log_s": out["rot_log_s"][k], "fingers": out["fingers"][k],
                       "fingers_log_s": out["fingers_log_s"][k], "pos": out["pos"][k], "log_size": out["log_size"][k],
                       "lost": bool(lost[k]), "score": float(score[k])}
        return hand

    def _head(self, obs):
        face = obs.get("face")
        if face is not None and len(face) >= 468:
            r = fit(self.canon, np.asarray(face)[:468, :3])
            return r["R"] @ self.face_offset.T @ self.rest_head.T, r["centre"], float(np.log(r["scale"])), True
        return np.eye(3), np.zeros(2), 0.0, False

    def _combine(self, body_L, hand, head_G, head_w):
        """`mttrain.merger.batch.combine` for one frame."""
        G = globals_from_locals(body_L, self.parent_idx)
        J = self.J
        for s in "lr":
            deg = np.degrees(np.exp(hand[s]["rot_log_s"]))
            w = np.clip((self.conf_deg[1] - deg) / (self.conf_deg[1] - self.conf_deg[0]), 0, 1)
            tw = G[J[f"{s}_wrist_twist"]]
            Gw = slerp_mat(G[J[f"{s}_wrist"]], hand[s]["G"], w)
            if np.degrees(geodesic(tw.T @ Gw, np.eye(3))) <= self.max_bend:
                G[J[f"{s}_wrist"]] = Gw
        G[J["c_head"]] = slerp_mat(G[J["c_head"]], head_G, head_w)
        return locals_from_globals(G, self.parent_idx)

    # --- one frame ----------------------------------------------------------------------------------------------
    def step(self, t, obs):
        """t: seconds; obs: Detector.detect output (or a recorder-format frame with arrays). Returns (bones: VRM
        bone -> quaternion xyzw, confidence: joint -> 0..1, present, root (3,) camera frame m, up (3,) camera frame)."""
        present, rot, log_s, root, root_ls, up = self._body(t, obs)
        hand = self._hands(t, obs)
        head_G, centre, log_scale, seen = self._head(obs)
        base = self._combine(rot, hand, head_G, self.head_w * float(seen))
        nj = self.nj
        bs = obs.get("blendshapes")
        sv = float(seen)
        parts = [to6d(rot[:nj]).reshape(-1), log_s[:nj], to6d(base[:nj]).reshape(-1), root, root_ls, up]
        for s in "lr":
            h = hand[s]
            parts += [to6d(h["G"]), [h["rot_log_s"]], h["fingers"], h["fingers_log_s"], h["pos"], [h["log_size"]],
                      [float(h["lost"])], [h["score"]]]
        parts += [to6d(head_G) * sv, centre * sv, [log_scale * sv], [sv],
                  [bs.get(n, 0.0) for n in self.blend_names] if bs else [0.0] * len(self.blend_names), [float(bool(bs))]]
        self.frames.insert(0, np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]))
        self.frame_times.insert(0, t)
        prune_history(self.frames, self.frame_times, t, max(self.offsets) * TRAIN_DT)
        # a step older than the whole history takes the oldest frame (the session's first)
        ages = shifted_ages([o * TRAIN_DT for o in self.offsets], self.history_shift)
        idx = [len(self.frames) - 1 if i is None else i for i in pick_history(self.frame_times, t, ages)]
        x = np.stack([self.frames[i] for i in idx])[None].astype(np.float32)
        names = [o.name for o in self.merger_net.get_outputs()]
        out = dict(zip(names, (v[0].astype(np.float64) for v in self.merger_net.run(None, {"x": x}))))
        L = locals_from_globals(globals_from_locals(base[:nj], self.parent_idx) @ out["res"], self.parent_idx)
        if nj < len(base):
            L = np.concatenate([L, base[nj:]], 0)
        # hip position and up (`mttrain.merger.model.apply_root`)
        r = out["root_res"]
        z0 = max(root[2], 0.1)
        z = z0 * np.exp(r[2])
        root_m = np.array([(root[0] / z0 + r[0]) * z, (root[1] / z0 + r[1]) * z, z])
        u = up + out["up_res"]
        up_m = u / max(np.linalg.norm(u), 1e-12)
        q = np.zeros((2, 27))
        for k, s in enumerate("lr"):
            q[k, self.out_idx] = hand[s]["fingers"] + out["fingers_res"][k]
        q = np.clip(q, self.lo, self.hi).reshape(1, -1)
        self.last = {"hand": hand, "out": out, "base": base, "fingers": q}  # for checks
        G = self.pose.mhr_globals(L[None], fingers=q)
        locals_ = self.pose.vrm_from_globals(G)
        if self.retarget_on and self.retarget is not None:
            locals_ = self.retarget(G, locals_)
        bones = {b: locals_[b][0] for b in self.bones if b in locals_}
        conf = dict(zip(self.joints, np.exp(-out["log_s"][:len(self.joints)]).tolist()))
        return bones, conf, present, root_m, up_m
