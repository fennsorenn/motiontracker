"""MediaPipe Pose's systematic biases, applied to geometrically exact synthetic landmarks.

Calibrated on the first recorded session (2026-11-01) against the training population (fitted CMU
subjects, not MHR's template); `python -m mttrain.calibrate` reprints the table for any sessions.
Ratios to shoulder width:

| | real world | real image | synthetic raw (world / image) |
|---|---|---|---|
| eye to eye | 0.10 | 0.23 | 0.17 / 0.19 |
| ear to ear | 0.45 | 0.42 | 0.48 / 0.49 |
| mouth width | 0.13 | 0.17 | 0.10 / 0.12 |
| upper arm | 0.62 | (pose-dependent) | 0.80 |
| forearm | 0.69 | (pose-dependent) | 0.76 |
| hip width | 0.66 | 0.65 | 0.51 / 0.47 |
| torso (shoulder to hip) | 1.37 | | 1.39 |
| nose to shoulder | 0.82 | | 0.78 |

Two kinds of bias follow:
- Placement (both spaces; the image factor): where MediaPipe puts a landmark on the body. Eyes and
  mouth corners sit wider, ears narrower, hip points much wider than MHR's hip joints. Applied in
  3D before projection.
- World-only (world factor / image factor): the world landmarks squeeze the eyes together, widen
  the ears slightly and shorten the arms; the image landmarks don't. Applied to world landmarks
  after the camera transform.

Image z (MediaPipe's relative depth in pose2d): on the session it is 0.84 x world z / shoulder width
(correlation 0.91, arm and face landmarks); geometrically exact synthetic image z is 0.26 x. MediaPipe
exaggerates its image depth about 3x, so synthetic image z is scaled by IMAGE_Z (found 2026-11-08:
without it the M1 model misread resting arms by 35-40 degrees on real input).

Depth flattening (found 2026-11-09 on rendered sessions with exact ground truth, mttrain.render):
MediaPipe's world landmarks compress depth. Per segment, the camera-depth component of the segment
vector is about gain x true + bias, while the segment length is kept (the image-plane part is
stretched). Gains 0.2-0.6 (upper arm 0.37, forearm 0.33, torso 0.42, head yaw 0.47), biases up to
-10 cm (torso leaning toward the camera, head forward). Image z is about 3x the world z on every
segment, so the synthetic image z is derived from the flattened depth (then x IMAGE_Z). Without the
flattening, synthetic depth was about 3x MediaPipe's in both spaces, and the model inherited the
flattening on real input. Not modelled yet: an extra depth error during motion (4-7 cm, about
0.5 s timescale; under 1.5 cm at rest).

Head rotation (found 2026-11-11 on rendered sessions, `python -m mttrain.headcal`): MediaPipe's
face points follow the head's rotation relative to the chest only partly, and noisily. Head-vs-chest
angles built the same way from MediaPipe's and from the true landmarks: world landmarks slope 0-0.28
(correlation <= 0.6), image landmarks slope ~0.4-0.75 (correlation ~0.5-0.75); the true landmarks
follow the skeleton with slope ~1. Synthetic face points rotate with the head exactly, so a model
trained on them under-reads the head on real input. Modelled per space (world, image) as a gain on
the head-vs-chest rotation (relative to MediaPipe's own shoulder/hip frame, HeadRig) plus slow rotation
noise, about the neck joint (head_lock); the 3D head
points are also dropped for some sequences (head3d), so the model can run with or without them.

One person, one session: every factor is randomized per sequence (log-normal) so the model can't
lean on these exact values. Recalibrate when more seated-v2 sessions exist.
"""

from dataclasses import dataclass, field

import numpy as np

from mtcore.landmarks import MP33

I = {n: k for k, n in enumerate(MP33)}
FACE = [I[n] for n in MP33[:11]]
UPPER = [k for k in range(33) if k not in (I[n] for n in MP33[23:])]  # everything above the hips


def _sym(d):
    out = dict(d)
    for n, f in d.items():
        out[n.replace("left", "right")] = f
    return out


# lateral factor on each face point's offset from the ear midpoint, along the ear-to-ear axis
FACE_PLACEMENT = _sym({"left_eye_inner": 1.2, "left_eye": 1.2, "left_eye_outer": 1.2, "mouth_left": 1.4,
                       "left_ear": 0.87})
FACE_WORLD = _sym({"left_eye_inner": 0.49, "left_eye": 0.49, "left_eye_outer": 0.49, "mouth_left": 0.89,
                   "left_ear": 1.09})
HIP_WIDEN = 1.4  # hip half-width factor (placement)
IMAGE_Z = 3.0  # MediaPipe image z / geometric image z (see docstring)
LIMB = {"upper_arm": 0.77, "forearm": 0.90, "torso": 0.99, "neck": 1.05}  # world only
# segment start, segment end, landmarks carried with the shortening
CHAINS = [
    ("upper_arm", "left_shoulder", "left_elbow", ["left_elbow", "left_wrist", "left_pinky", "left_index", "left_thumb"]),
    ("upper_arm", "right_shoulder", "right_elbow", ["right_elbow", "right_wrist", "right_pinky", "right_index", "right_thumb"]),
    ("forearm", "left_elbow", "left_wrist", ["left_wrist", "left_pinky", "left_index", "left_thumb"]),
    ("forearm", "right_elbow", "right_wrist", ["right_wrist", "right_pinky", "right_index", "right_thumb"]),
]

# depth flattening: segment -> (gain, bias in metres) on the camera-depth component; see docstring
DEPTH = {"torso": (0.42, -0.097), "shoulders": (0.60, 0.010), "hips": (0.47, -0.006), "neck": (0.25, -0.039),
         "ears": (0.47, 0.004), "face": (0.43, -0.047), "upper_arm": (0.37, 0.009), "forearm": (0.33, -0.057),
         "hand": (0.18, -0.041)}
DEPTH_KEYS = list(DEPTH)
# perspective: gain change per metre of camera distance (FOV experiment 2026-11-09: the same motion at
# 50-95 deg with constant framing, 0.64-1.5 m; closer cameras flatten the arms, torso and neck more and
# the left-right depth slightly less); DEPTH holds the gains at DEPTH_REF_DIST. Face and hips: not measured.
DEPTH_SLOPE = {"torso": 0.13, "shoulders": -0.12, "hips": 0.0, "neck": 0.20, "ears": -0.10, "face": 0.0,
               "upper_arm": 0.14, "forearm": 0.24, "hand": 0.09}
DEPTH_REF_DIST = 1.1  # metres, camera to shoulders (median of the calibration renders)

# head rotation (see docstring): gain on the head-vs-chest rotation and slow rotation noise (std deg,
# time constant s), per space. Calibrated 2026-11-11 with mttrain.headcal on the four rendered sets
# (pooled slope/corr turn, nod, tilt; MediaPipe vs synthetic): image 0.54/0.54, 0.34/0.40, 0.38/0.40 vs
# 0.57/0.53, 0.52/0.50, 0.34/0.36; world 0.07/0.18, 0.12/0.18, 0.09/0.09 vs 0.11/0.06, 0.08/0.05,
# 0.08/0.07 (world correlation is limited by the general landmark jitter; image nod also comes from
# placement and flattening, so it stays high). The renders' faces are untextured; real faces may
# respond more, which the per-sequence spread (Style.sample head_spread) is meant to cover.
HEAD_GAIN = {"world": 0.1, "image": 0.25}
HEAD_NOISE_DEG = {"world": 1.0, "image": 8.0}
HEAD_NOISE_TAU = 0.5
HEAD3D_DROP = 0.3  # fraction of sequences without the 3D head points (world xyz, image z)


def _depth_steps():
    """(key, start points, end points, moved indices, weights): applied in order; each step only
    translates the points below it, so every segment's final vector depends on its own step only."""
    i = lambda *ns: [I[n] for n in ns]
    arm = lambda s: i(f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist", f"{s}_pinky", f"{s}_index", f"{s}_thumb")
    leg = lambda s: i(f"{s}_hip", f"{s}_knee", f"{s}_ankle", f"{s}_heel", f"{s}_foot_index")
    lface = i("left_eye_inner", "left_eye", "left_eye_outer", "mouth_left")
    rface = i("right_eye_inner", "right_eye", "right_eye_outer", "mouth_right")
    ears = i("left_ear", "right_ear")
    hips, shoulders = i("left_hip", "right_hip"), i("left_shoulder", "right_shoulder")
    steps = [("torso", hips, shoulders, UPPER, [1.0] * len(UPPER)),
             ("shoulders", i("right_shoulder"), i("left_shoulder"), arm("left") + arm("right"), [0.5] * 6 + [-0.5] * 6),
             ("hips", i("right_hip"), i("left_hip"), leg("left") + leg("right"), [0.5] * 5 + [-0.5] * 5),
             ("neck", shoulders, ears, FACE, [1.0] * len(FACE)),
             ("ears", i("right_ear"), i("left_ear"), ears, [0.5, -0.5])]
    steps += [("face", ears, [p], [p], [1.0]) for p in i("nose") + lface + rface]
    for s in ("left", "right"):
        steps += [("upper_arm", i(f"{s}_shoulder"), i(f"{s}_elbow"), arm(s)[1:], [1.0] * 5),
                  ("forearm", i(f"{s}_elbow"), i(f"{s}_wrist"), arm(s)[2:], [1.0] * 4)]
        steps += [("hand", i(f"{s}_wrist"), [p], [p], [1.0]) for p in i(f"{s}_pinky", f"{s}_index", f"{s}_thumb")]
    return steps


def rotvec_to_mat(v):
    """Rotation vectors (..., 3) -> matrices (..., 3, 3) (Rodrigues)."""
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(th, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rotvec(M):
    """Rotation matrices (..., 3, 3) -> rotation vectors (..., 3) (angles below pi)."""
    c = np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    w = np.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)
    s = np.sin(th)
    return w * np.where(s > 1e-6, th / np.maximum(2 * s, 1e-12), 0.5)[..., None]


def head_lock(X, delta, pivot, gain, noise):
    """MediaPipe's partial head-rotation response on landmarks X (T, 33, 3), any frame/unit of pivot.
    delta (T, 3, 3): the head's rotation relative to the chest, in X's frame (world rotation taking the
    head as if rigid with the chest to the actual head); pivot (T, 3): the neck joint; noise (T, 3)
    rotation vectors. The face points are turned as if the head had rotated by noise * delta^gain."""
    want = rotvec_to_mat(noise) @ rotvec_to_mat(gain * mat_to_rotvec(delta))
    M = want @ np.swapaxes(delta, -1, -2)
    X = X.copy()
    X[..., FACE, :] = pivot[..., None, :] + np.einsum("tij,tkj->tki", M, X[..., FACE, :] - pivot[..., None, :])
    return X


def landmark_frames(X):
    """Chest and head frames (..., 3, 3) (columns x, y, z) from landmarks (..., 33, 3): chest from the
    shoulders and hips, head from the ears, eyes and mouth corners."""
    def fr(xa, up):
        x = xa / np.linalg.norm(xa, axis=-1, keepdims=True)
        z = np.cross(x, up)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        return np.stack([x, np.cross(z, x), z], -1)
    mean = lambda *ns: X[..., [I[n] for n in ns], :].mean(-2)
    C = fr(X[..., I["left_shoulder"], :] - X[..., I["right_shoulder"], :],
           mean("left_shoulder", "right_shoulder") - mean("left_hip", "right_hip"))
    H = fr(X[..., I["left_ear"], :] - X[..., I["right_ear"], :],
           mean("left_eye", "right_eye") - mean("mouth_left", "mouth_right"))
    return C, H


class HeadRig:
    """The head's rotation relative to the chest as MediaPipe's points see it: landmark frames (the
    chest from shoulders and hips; MediaPipe ties its face points to those, not to the spine), relative
    to the rest pose. Returns delta (T, 3, 3), the world rotation taking the face as if rigid with the
    chest frame to the actual face, and the neck joint (T, 3) in metres (the pivot)."""

    def __init__(self, joint_names, rest_state, landmark_model):
        self.n = list(joint_names).index("c_neck")
        C0, H0 = landmark_frames(landmark_model(rest_state[None])[0] / 100.0)
        self.rest_rel = C0.T @ H0  # chest -> head at rest

    def __call__(self, states, X):
        """states (T, J, 8); X (T, 33, 3) the true landmarks in metres (before any style)."""
        C, H = landmark_frames(X)
        delta = H @ self.rest_rel.T @ np.swapaxes(C, -1, -2)
        return delta, states[..., self.n, :3] / 100.0


def head_noise(rng, t, sigma_deg, tau=HEAD_NOISE_TAU):
    """Slow rotation noise (T, 3) rad at irregular times t (T,) s: AR(1), stationary std sigma per axis."""
    out = np.zeros((len(t), 3))
    s = np.radians(sigma_deg)
    out[0] = rng.normal(0, s, 3)
    for i in range(1, len(t)):
        r = np.exp(-(t[i] - t[i - 1]) / tau)
        out[i] = r * out[i - 1] + np.sqrt(1 - r * r) * rng.normal(0, s, 3)
    return out


def _factors(d):
    f = np.ones(33)
    for n, v in d.items():
        f[I[n]] = v
    return f


def _lateral(X, factors):
    """Scale each face point's offset from the ear midpoint along the ear-to-ear axis."""
    ears = X[..., I["left_ear"], :] - X[..., I["right_ear"], :]
    axis = ears / np.maximum(np.linalg.norm(ears, axis=-1, keepdims=True), 1e-9)
    centre = 0.5 * (X[..., I["left_ear"], :] + X[..., I["right_ear"], :])
    lat = np.einsum("...kc,...c->...k", X[..., FACE, :] - centre[..., None, :], axis)
    X[..., FACE, :] += ((factors[FACE] - 1) * lat)[..., None] * axis[..., None, :]
    return X


DEPTH_STEPS = _depth_steps()


FLAT_MAX_STRETCH = 4.0  # cap on the image-plane stretch (segments pointing nearly at the camera)


def flatten(W, gain, bias):
    """MediaPipe's depth flattening on hip-centred camera-frame landmarks W (..., 33, 3), metres.
    gain, bias: (K,) per DEPTH_KEYS (bias in metres). Per step, the segment's depth component becomes
    gain * depth + bias and its image-plane part stretches (direction kept) so the length stays."""
    W = W.copy()
    for key, a, b, moved, w in DEPTH_STEPS:
        k = DEPTH_KEYS.index(key)
        d = W[..., b, :].mean(-2) - W[..., a, :].mean(-2)
        L = np.linalg.norm(d, axis=-1)
        ez = np.clip(gain[k] * d[..., 2] + bias[k], -L, L)
        xy = np.linalg.norm(d[..., :2], axis=-1)
        s = np.minimum(np.sqrt(np.maximum(L * L - ez * ez, 0.0)) / np.maximum(xy, 1e-9), FLAT_MAX_STRETCH)
        e = np.concatenate([d[..., :2] * s[..., None], ez[..., None]], -1)
        W[..., moved, :] += np.asarray(w)[:, None] * (e - d)[..., None, :]
    return W


@dataclass
class Style:
    hip_widen: float = 1.0
    image_z: float = 1.0
    face_placement: np.ndarray = field(default_factory=lambda: np.ones(33))
    face_world: np.ndarray = field(default_factory=lambda: np.ones(33))
    limb: dict = field(default_factory=lambda: {k: 1.0 for k in LIMB})
    offset: np.ndarray = field(default_factory=lambda: np.zeros((33, 3)))  # world metres, per landmark
    depth_gain: np.ndarray = field(default_factory=lambda: np.ones(len(DEPTH)))  # per DEPTH_KEYS
    depth_bias: np.ndarray = field(default_factory=lambda: np.zeros(len(DEPTH)))  # metres
    depth_relax: float = 0.0  # 0: render-calibrated flattening, 1: none (see depth_at)
    head_gain: dict = field(default_factory=lambda: {"world": 1.0, "image": 1.0})  # head_lock gain per space
    head_noise_deg: dict = field(default_factory=lambda: {"world": 0.0, "image": 0.0})
    head3d: bool = True  # False: the 3D head points are dropped from the model input

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def calibrated(cls):
        return cls(HIP_WIDEN, IMAGE_Z, _factors(FACE_PLACEMENT), _factors(FACE_WORLD), dict(LIMB),
                   depth_gain=np.array([g for g, _ in DEPTH.values()]), depth_bias=np.array([b for _, b in DEPTH.values()]),
                   head_gain=dict(HEAD_GAIN), head_noise_deg=dict(HEAD_NOISE_DEG))

    @classmethod
    def sample(cls, rng, spread=0.08, offset_cm=0.8, image_z_spread=0.25, depth_spread=0.25, head_spread=0.4):
        """Calibrated values with per-sequence log-normal spread (face factors shared, so it stays symmetric).
        Image z gets a wider spread: its scale is the least certain and may differ between MediaPipe models.
        Depth gains: log-normal depth_spread, clipped to [0.05, 1]; biases: normal, 1 cm + 30% of the bias;
        depth_relax uniform in [0, 1]: real video may be flattened less than the renders it was calibrated
        on (2026-11-09), so the strength is left for the model to infer.
        Head: gains log-normal head_spread (clipped to [0.02, 1]), noise std log-normal head_spread; the
        3D head points are dropped for HEAD3D_DROP of the sequences."""
        ln = lambda: float(np.exp(rng.normal(0, spread)))
        fp, fw = _factors(FACE_PLACEMENT), _factors(FACE_WORLD)
        fp[FACE] = 1 + (fp[FACE] - 1) * ln()
        fw[FACE] *= ln()
        g0, b0 = np.array([g for g, _ in DEPTH.values()]), np.array([b for _, b in DEPTH.values()])
        return cls(HIP_WIDEN * ln(), IMAGE_Z * float(np.exp(rng.normal(0, image_z_spread))), fp, fw, {k: v * ln() for k, v in LIMB.items()},
                   rng.normal(0, offset_cm / 100, (33, 3)),
                   np.clip(g0 * np.exp(rng.normal(0, depth_spread, len(g0))), 0.05, 1.0),
                   b0 + rng.normal(0, 1, len(b0)) * (0.01 + 0.3 * np.abs(b0)), float(rng.uniform(0, 1)),
                   {k: float(np.clip(g * np.exp(rng.normal(0, head_spread)), 0.02, 1.0)) for k, g in HEAD_GAIN.items()},
                   {k: float(v * np.exp(rng.normal(0, head_spread))) for k, v in HEAD_NOISE_DEG.items()},
                   bool(rng.random() >= HEAD3D_DROP))

    def placement(self, X):
        """3D landmarks (..., 33, 3), any frame and unit: where MediaPipe places its points."""
        X = _lateral(X.copy(), self.face_placement)
        mid = 0.5 * (X[..., I["left_hip"], :] + X[..., I["right_hip"], :])
        for n in ("left_hip", "right_hip"):
            X[..., I[n], :] = mid + self.hip_widen * (X[..., I[n], :] - mid)
        return X

    def depth_at(self, dist):
        """Effective (gain, bias) per DEPTH_KEYS for a camera `dist` metres from the shoulders: the
        perspective slope, then blended toward no flattening by depth_relax."""
        slope = np.array([DEPTH_SLOPE[k] for k in DEPTH_KEYS])
        g = np.clip(self.depth_gain + slope * (dist - DEPTH_REF_DIST), 0.02, 1.0)
        return g + self.depth_relax * (1 - g), self.depth_bias * (1 - self.depth_relax)

    def flatten(self, W, dist=DEPTH_REF_DIST):
        """Hip-centred camera-frame landmarks (..., 33, 3), metres -> MediaPipe's flattened depth (both
        spaces), for a camera `dist` metres from the shoulders."""
        return flatten(W, *self.depth_at(dist))

    def world(self, W):
        """World landmarks (..., 33, 3), metres, hip-midpoint origin, already flattened: MediaPipe's
        world-only distortions."""
        W = W.copy()
        # torso: everything above the hips moves along the hip-to-shoulder-midpoint vector
        upper = UPPER
        sh_mid = 0.5 * (W[..., I["left_shoulder"], :] + W[..., I["right_shoulder"], :])
        W[..., upper, :] -= ((1 - self.limb["torso"]) * sh_mid)[..., None, :]
        # neck: face along the shoulder-midpoint-to-nose vector
        sh_mid = 0.5 * (W[..., I["left_shoulder"], :] + W[..., I["right_shoulder"], :])
        W[..., FACE, :] -= ((1 - self.limb["neck"]) * (W[..., I["nose"], :] - sh_mid))[..., None, :]
        W = _lateral(W, self.face_world)
        for seg, a, b, moved in CHAINS:
            d = W[..., I[b], :] - W[..., I[a], :]
            W[..., [I[m] for m in moved], :] -= ((1 - self.limb[seg]) * d)[..., None, :]
        return W + self.offset
