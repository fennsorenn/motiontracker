"""Head pose from MediaPipe's face mesh: a weighted similarity fit of MediaPipe's canonical face model to the face
landmarks, as MediaPipe's own face geometry pipeline does (same canonical mesh, same 33 weighted Procrustes points),
but in image units (weak perspective) instead of MediaPipe's metric camera model.

The canonical model comes from the Face Landmarker bundle the runtime already downloads (Apache-2.0,
`geometry_pipeline_metadata_landmarks.binarypb`; docs/LICENSES-REGISTER.md); it is read here with a minimal
protobuf wire decoder so the base install stays numpy-only.

Image points: Holistic face landmarks (x, y normalized over the frame, z relative depth on roughly the scale of x),
as the recorder and the runtime report them (mirrored image; `mttrain.camera`: an unmirrored camera on the performer
reproduces them). They are turned into a camera frame in units of the frame width: x right, y down, z away.

Result per frame: rotation R (3, 3) camera <- face, R = identity for a face looking straight into the camera
upright; scale (frame widths per canonical cm); centre (x, y in frame widths from the image centre, the fit's
translation); rms residual (frame widths).
"""

import struct
import zipfile
from pathlib import Path

import numpy as np

METADATA_ENTRY = "geometry_pipeline_metadata_landmarks.binarypb"
FACE_BUNDLE = "face_landmarker-float16-1.task"
ASPECT = 4 / 3
# canonical model axes (x across the face, y up, z out of the face) -> camera-like axes (y down, z away):
# a face looking into the camera then has rotation identity
FLIP = np.diag([1.0, -1.0, -1.0])


def _varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _fields(b):
    """Protobuf message bytes -> list of (field number, wire type, value: int or bytes or float bytes)."""
    out, i = [], 0
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.append((f, wt, v))
    return out


def _packed_floats(chunks):
    vals = []
    for wt, v in chunks:
        vals.extend(struct.unpack(f"<{len(v) // 4}f", v) if wt == 2 else struct.unpack("<f", v))
    return np.array(vals)


class CanonicalFace:
    """MediaPipe's canonical face: vertices (468, 3) in cm, Procrustes landmark ids (33,) and weights (33,)."""

    def __init__(self, bundle=None):
        if bundle is None:
            import os
            bundle = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "motiontracker" / FACE_BUNDLE
        meta = zipfile.ZipFile(bundle).read(METADATA_ENTRY)
        # GeometryPipelineMetadata: canonical_mesh = 1 (Mesh3d: vertex_type 1, primitive_type 2,
        # vertex_buffer 3 (x, y, z, u, v per vertex), index_buffer 4), procrustes_landmark_basis = 2
        # (WeightedLandmarkRef: landmark_id 1, weight 2), input_source = 3
        ids, weights, verts = [], [], None
        for f, _, v in _fields(meta):
            if f == 1:
                vb = [(wt, x) for g, wt, x in _fields(v) if g == 3]
                verts = _packed_floats(vb).reshape(-1, 5)[:, :3]
            elif f == 2:
                ref = {g: x for g, _, x in _fields(v)}
                ids.append(ref.get(1, 0))
                weights.append(struct.unpack("<f", ref[2])[0] if 2 in ref else 0.0)
        self.vertices = verts
        self.ids = np.array(ids)
        self.weights = np.array(weights)
        self.points = verts[self.ids] @ FLIP.T  # the Procrustes points in camera-like axes

    @classmethod
    def from_arrays(cls, points, ids, weights):
        """From the Procrustes points (camera-like axes), ids and weights alone (merger bundles carry these)."""
        c = cls.__new__(cls)
        c.vertices = None
        c.points = np.asarray(points, dtype=np.float64)
        c.ids, c.weights = np.asarray(ids), np.asarray(weights, dtype=np.float64)
        return c


def image_points(face, aspect=ASPECT):
    """Face landmarks (..., >=468, 3) normalized -> camera-frame points in frame widths, origin at the image centre."""
    face = np.asarray(face, dtype=np.float64)
    return np.stack([face[..., 0] - 0.5, (face[..., 1] - 0.5) / aspect, face[..., 2]], -1)


def fit(canon, face, aspect=ASPECT):
    """Face landmarks (..., >=468, 3) -> dict R (..., 3, 3), scale (...), centre (..., 2), rms (...): weighted
    similarity fit (Umeyama) of the canonical Procrustes points to the observed ones."""
    P = image_points(face, aspect)[..., canon.ids, :]
    C = canon.points
    w = canon.weights / canon.weights.sum()
    mp_ = (w[:, None] * P).sum(-2)
    mc = (w[:, None] * C).sum(0)
    Pc, Cc = P - mp_[..., None, :], C - mc
    S = np.einsum("k,...ki,kj->...ij", w, Pc, Cc)  # cross-covariance (..., 3, 3)
    U, sig, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(sig.shape)
    D[..., -1] = d
    R = U @ (D[..., :, None] * Vt)
    var_c = (w * (Cc ** 2).sum(-1)).sum()
    scale = (sig * D).sum(-1) / var_c
    t = mp_ - scale[..., None] * np.einsum("...ij,j->...i", R, mc)
    res = P - (scale[..., None, None] * np.einsum("...ij,kj->...ki", R, C) + t[..., None, :])
    rms = np.sqrt((w * (res ** 2).sum(-1)).sum(-1))
    return {"R": R, "scale": scale, "centre": t[..., :2], "rms": rms}
