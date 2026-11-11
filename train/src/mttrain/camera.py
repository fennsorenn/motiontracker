"""Synthetic webcam: MHR-space landmarks -> what MediaPipe Pose reports (before noise).

Conventions, measured on recorded sessions (vspark mirrors the image before detection):
- Labels describe the person in the (mirrored) image, so synthesizing an ordinary, unmirrored
  camera looking at the MHR performer reproduces them directly; the avatar then mirrors the user,
  as the baseline does.
- World landmarks: metres, origin at the hip midpoint, camera-aligned axes (x image-right, y down,
  z away from the camera).
- Image landmarks (`pose2d`): x, y normalized to [0, 1] over the frame; z is depth relative to the
  hips on roughly the scale of x.
"""

from dataclasses import dataclass

import numpy as np

from mtcore.landmarks import MP33

L_SH, R_SH = MP33.index("left_shoulder"), MP33.index("right_shoulder")
L_HIP, R_HIP = MP33.index("left_hip"), MP33.index("right_hip")


@dataclass
class Camera:
    R: np.ndarray  # (3, 3) world (MHR, metres) -> camera axes (x right, y down, z forward)
    t: np.ndarray  # (3,) camera position in MHR world, metres
    fx: float  # focal length in units of image width
    aspect: float = 4 / 3  # width / height

    def to_camera(self, X):
        """MHR world points (..., 3) in metres -> camera frame."""
        return (X - self.t) @ self.R.T

    def project(self, X):
        """-> (pose2d (..., 3) as MediaPipe reports it, in_front mask)."""
        C = self.to_camera(X)
        z = np.maximum(C[..., 2], 1e-3)
        u = 0.5 + self.fx * C[..., 0] / z
        v = 0.5 + self.fx * self.aspect * C[..., 1] / z
        hip_z = 0.5 * (z[..., L_HIP] + z[..., R_HIP])[..., None]
        d = self.fx * (C[..., 2] - hip_z) / hip_z
        return np.stack([u, v, d], -1), C[..., 2] > 0.05

    def flat_image_z(self, X, flat):
        """pose2d z from hip-centred camera-frame depths `flat` (..., 33, 3) instead of the true ones
        (project's z is fx * (z - hip_z) / hip_z, and world_landmarks' z is z - hip_z)."""
        C = self.to_camera(X)
        hip_z = np.maximum(0.5 * (C[..., L_HIP, 2] + C[..., R_HIP, 2]), 1e-3)[..., None]
        return self.fx * flat[..., 2] / hip_z

    def world_landmarks(self, X):
        """MediaPipe world landmarks: camera-aligned, metres, hip-midpoint origin."""
        C = self.to_camera(X)
        return C - 0.5 * (C[..., L_HIP, :] + C[..., R_HIP, :])[..., None, :]


def look_at(eye, target, roll=0.0):
    """Camera rotation for MHR world (y up): camera z toward target, camera y down."""
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])  # image-right
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)  # image-down
    c, s = np.cos(roll), np.sin(roll)
    r, d = c * r + s * d, -s * r + c * d
    return np.stack([r, d, f])


FOV_RANGE = (50.0, 130.0)  # horizontal, degrees (owner's cameras: about 70 and 90)
DIST_RANGE = (0.45, 1.6)  # camera to shoulders, metres (log-uniform)
SW_RANGE = (0.10, 0.50)  # shoulder width / frame width: distance is clamped to keep the framing in it


def sample_desk_camera(rng, landmarks_m, fov=None):
    """Desk-webcam camera for a clip, from its landmarks (T, 33, 3) in MHR metres.

    Field of view, distance and framing vary: horizontal FOV uniform in FOV_RANGE, distance to the
    shoulders log-uniform in DIST_RANGE but clamped so the shoulder width covers SW_RANGE of the frame
    width, shoulders at 40-75% of the frame height; plus random yaw, camera height above the chest,
    roll. The camera faces the performer's average front.
    `fov` (horizontal, degrees) overrides the sampled one; the random stream is unchanged, so the
    same rng state gives the same camera apart from field of view and distance: the camera moves along
    the same line from the chest (height scaled with distance), so the viewing angle and the framing
    stay and only the perspective changes."""
    sh = landmarks_m[:, [L_SH, R_SH]].mean(1)  # (T, 3)
    chest = np.median(sh, 0)
    across = np.median(landmarks_m[:, L_SH] - landmarks_m[:, R_SH], 0)  # toward the person's left
    width = float(np.linalg.norm(across))
    front = np.cross(across, [0.0, 1.0, 0.0])
    front[1] = 0.0
    front /= np.linalg.norm(front)
    yaw = np.radians(rng.uniform(-25, 25))
    c, s = np.cos(yaw), np.sin(yaw)
    dirn = np.array([c * front[0] + s * front[2], 0.0, -s * front[0] + c * front[2]])
    fov_sampled = rng.uniform(*FOV_RANGE)
    fx_s = 0.5 / np.tan(np.radians(fov_sampled) / 2)
    dist = float(np.exp(rng.uniform(np.log(DIST_RANGE[0]), np.log(DIST_RANGE[1]))))
    dist = float(np.clip(dist, fx_s * width / SW_RANGE[1], fx_s * width / SW_RANGE[0]))
    fx = fx_s if fov is None else 0.5 / np.tan(np.radians(fov) / 2)
    dist *= fx / fx_s  # same framing at an overridden fov
    sh_y = rng.uniform(0.40, 0.75)  # shoulder height in the frame (0 = top)
    lift = rng.uniform(0.0, 0.55) * fx / fx_s
    eye = chest + dist * dirn + [0.0, lift, 0.0]
    # aim above or below the chest so the shoulders land at sh_y
    target = chest + [0.0, dist * (sh_y - 0.5) / (fx * 4 / 3), 0.0]
    return Camera(R=look_at(eye, target, np.radians(rng.uniform(-4, 4))), t=eye, fx=fx)


def shoulder_distance(cam, X):
    """Median camera-frame depth of the shoulder midpoint over a clip's landmarks X (T, 33, 3), metres."""
    return float(np.median(cam.to_camera(X)[:, [L_SH, R_SH], 2].mean(1)))
