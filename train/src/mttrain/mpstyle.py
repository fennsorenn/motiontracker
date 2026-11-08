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

One person, one session: every factor is randomized per sequence (log-normal) so the model can't
lean on these exact values. Recalibrate when more seated-v2 sessions exist.
"""

from dataclasses import dataclass, field

import numpy as np

from mtcore.landmarks import MP33

I = {n: k for k, n in enumerate(MP33)}
FACE = [I[n] for n in MP33[:11]]


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


@dataclass
class Style:
    hip_widen: float = 1.0
    image_z: float = 1.0
    face_placement: np.ndarray = field(default_factory=lambda: np.ones(33))
    face_world: np.ndarray = field(default_factory=lambda: np.ones(33))
    limb: dict = field(default_factory=lambda: {k: 1.0 for k in LIMB})
    offset: np.ndarray = field(default_factory=lambda: np.zeros((33, 3)))  # world metres, per landmark

    @classmethod
    def identity(cls):
        return cls()

    @classmethod
    def calibrated(cls):
        return cls(HIP_WIDEN, IMAGE_Z, _factors(FACE_PLACEMENT), _factors(FACE_WORLD), dict(LIMB))

    @classmethod
    def sample(cls, rng, spread=0.08, offset_cm=0.8, image_z_spread=0.25):
        """Calibrated values with per-sequence log-normal spread (face factors shared, so it stays symmetric).
        Image z gets a wider spread: its scale is the least certain and may differ between MediaPipe models."""
        ln = lambda: float(np.exp(rng.normal(0, spread)))
        fp, fw = _factors(FACE_PLACEMENT), _factors(FACE_WORLD)
        fp[FACE] = 1 + (fp[FACE] - 1) * ln()
        fw[FACE] *= ln()
        return cls(HIP_WIDEN * ln(), IMAGE_Z * float(np.exp(rng.normal(0, image_z_spread))), fp, fw, {k: v * ln() for k, v in LIMB.items()},
                   rng.normal(0, offset_cm / 100, (33, 3)))

    def placement(self, X):
        """3D landmarks (..., 33, 3), any frame and unit: where MediaPipe places its points."""
        X = _lateral(X.copy(), self.face_placement)
        mid = 0.5 * (X[..., I["left_hip"], :] + X[..., I["right_hip"], :])
        for n in ("left_hip", "right_hip"):
            X[..., I[n], :] = mid + self.hip_widen * (X[..., I[n], :] - mid)
        return X

    def world(self, W):
        """World landmarks (..., 33, 3), metres, hip-midpoint origin: MediaPipe's world-space distortions."""
        W = W.copy()
        # torso: everything above the hips moves along the hip-to-shoulder-midpoint vector
        upper = [k for k in range(33) if k not in (I[n] for n in MP33[23:])]
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
