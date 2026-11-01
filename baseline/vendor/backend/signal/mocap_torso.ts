import { Quaternion } from '@vspark/shared/signal';

/**
 * Torso frame shared by the MediaPipe converters.
 *
 * pose_torso_head_to_bones writes the chest orientation (split across spine + chest) and
 * pose_arms_to_bones expresses the upper arms relative to that same chest. Both must agree on the
 * exact chest rotation — including the yaw damping — or every arm comes out rotated by the
 * difference. Keeping the solve in one place makes that agreement structural.
 *
 * Landmarks passed here are already in the avatar frame (see `flipYZ` in the converters):
 * +X = subject's left, +Y = up, +Z = toward the camera.
 */

type Landmark = { x: number; y: number; z: number; visibility?: number };
type V3 = [number, number, number];

const VIS = 0.5;
const ok = (lm: Landmark): boolean => (lm.visibility ?? 1) >= VIS;

function sub(a: Landmark, b: Landmark): V3 {
  return [a.x - b.x, a.y - b.y, a.z - b.z];
}
function norm(v: V3): V3 {
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return l < 1e-9 ? [0, 1, 0] : [v[0] / l, v[1] / l, v[2] / l];
}
function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: V3, b: V3): V3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function mid(a: Landmark, b: Landmark): Landmark {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// frameToQuat(rightTarget, upTarget):
//
// Builds the quaternion R such that:
//   R × [1,0,0] = rightTarget   (VRM +X axis, exact)
//   R × [0,1,0] ≈ upTarget      (VRM +Y axis, Gram-Schmidt orthogonalised)
//   R × [0,0,-1] = derived      (VRM forward = -Z, fully derived)
//
// Using rightTarget as primary (exact) is optimal for the head because
// ear-to-ear gives the most reliable lateral axis.
//
// Uses Shepperd's method on the column-major rotation matrix.
// Column 0 = rightTarget (image of [1,0,0])
// Column 1 = upOrtho     (image of [0,1,0])
// Column 2 = -forward    (image of [0,0,1] = back of head)
// ─────────────────────────────────────────────────────────────────────────────
export function frameToQuat(rightTarget: V3, upTarget: V3): Quaternion {
  const X = norm(rightTarget);
  // Orthogonalise upTarget against X
  const d = dot(upTarget, X);
  const upOrtho: V3 = [
    upTarget[0] - X[0] * d,
    upTarget[1] - X[1] * d,
    upTarget[2] - X[2] * d,
  ];
  const Y = norm(upOrtho);
  // col2 = image of [0,0,1] (VRM back = away from camera in world = -Z world).
  // cross(Y, X) = up × right = -Z in right-hand convention.
  const col2 = norm(cross(X, Y));

  // Column-major matrix: col0=X, col1=Y, col2=col2
  // Shepperd for column-major:
  //   trace = X[0] + Y[1] + Zneg[2]
  //   qx = (Y[2]    - Zneg[1]) / 4qw
  //   qy = (Zneg[0] - X[2]   ) / 4qw
  //   qz = (X[1]    - Y[0]   ) / 4qw
  const trace = X[0] + Y[1] + col2[2];
  let qx: number, qy: number, qz: number, qw: number;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    qw = 0.25 / s;
    qx = (Y[2] - col2[1]) * s;
    qy = (col2[0] - X[2]) * s;
    qz = (X[1] - Y[0]) * s;
  } else if (X[0] > Y[1] && X[0] > col2[2]) {
    const s = 2 * Math.sqrt(1 + X[0] - Y[1] - col2[2]);
    qw = (Y[2] - col2[1]) / s;
    qx = 0.25 * s;
    qy = (Y[0] + X[1]) / s;
    qz = (col2[0] + X[2]) / s;
  } else if (Y[1] > col2[2]) {
    const s = 2 * Math.sqrt(1 + Y[1] - X[0] - col2[2]);
    qw = (col2[0] - X[2]) / s;
    qx = (Y[0] + X[1]) / s;
    qy = 0.25 * s;
    qz = (col2[1] + Y[2]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + col2[2] - X[0] - Y[1]);
    qw = (X[1] - Y[0]) / s;
    qx = (col2[0] + X[2]) / s;
    qy = (col2[1] + Y[2]) / s;
    qz = 0.25 * s;
  }
  const l = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  return new Quaternion(qx / l, qy / l, qz / l, qw / l);
}
// Decompose a unit quaternion into XYZ Euler angles (intrinsic, applied in order X then Y then Z).
// Used to scale each head-rotation axis independently for gain calibration.
export function quatToEulerXYZ(q: Quaternion): {
  x: number;
  y: number;
  z: number;
} {
  // Standard rotation-matrix-from-quaternion derivation, then extract XYZ Eulers.
  const x = q.x,
    y = q.y,
    z = q.z,
    w = q.w;
  const m11 = 1 - 2 * (y * y + z * z);
  const m12 = 2 * (x * y - z * w);
  const m13 = 2 * (x * z + y * w);
  const m23 = 2 * (y * z - x * w);
  const m33 = 1 - 2 * (x * x + y * y);
  // Y rotation comes from arcsin(m13). Clamp to avoid NaN at the gimbal-lock boundaries.
  const sy = Math.max(-1, Math.min(1, m13));
  const ey = Math.asin(sy);
  let ex: number, ez: number;
  if (Math.abs(m13) < 0.9999) {
    ex = Math.atan2(-m23, m33);
    ez = Math.atan2(-m12, m11);
  } else {
    ex = Math.atan2(2 * (y * z + x * w), 1 - 2 * (x * x + z * z));
    ez = 0;
  }
  return { x: ex, y: ey, z: ez };
}

export function eulerXYZToQuat(ex: number, ey: number, ez: number): Quaternion {
  const cx = Math.cos(ex / 2),
    sx = Math.sin(ex / 2);
  const cy = Math.cos(ey / 2),
    sy = Math.sin(ey / 2);
  const cz = Math.cos(ez / 2),
    sz = Math.sin(ez / 2);
  // XYZ order: q = qz * qy * qx
  return new Quaternion(
    sx * cy * cz + cx * sy * sz,
    cx * sy * cz - sx * cy * sz,
    cx * cy * sz + sx * sy * cz,
    cx * cy * cz - sx * sy * sz
  );
}

// Scale only the yaw (Y) component of a rotation, leaving pitch/roll intact. Used to damp the
// torso's turn: MediaPipe's shoulder estimate yaws along with a head turn, which would otherwise
// spill into the chest. Because the neck is computed relative to this same damped torso, the head
// keeps its true world orientation — the spilled yaw just moves from the chest into the neck.
// Now that the head is sourced from the face mesh (independent of the shoulders), the chest no
// longer needs heavy yaw damping to avoid head-turn spill — so allow most of the torso twist
// through. Note MediaPipe's shoulder-depth estimate is weak, so torso yaw reads softly regardless.
export const TORSO_YAW_GAIN = 0.7;
function dampYaw(q: Quaternion, gain: number): Quaternion {
  const e = quatToEulerXYZ(q);
  return eulerXYZToQuat(e.x, e.y * gain, e.z);
}
/**
 * World (avatar-frame) chest orientation from the shoulders and hips, with the yaw damping
 * applied. Shoulders must be visible (callers check); hips are optional — without them the
 * spine is assumed upright.
 */
export function torsoWorldQuat(
  ls: Landmark,
  rs: Landmark,
  lh: Landmark,
  rh: Landmark
): Quaternion {
  // sub(ls, rs) points toward the subject's left, i.e. VRM +X.
  const shdRight = norm(sub(ls, rs));
  let spineUp: V3;
  if (ok(lh) && ok(rh)) {
    spineUp = norm(sub(mid(ls, rs), mid(lh, rh)));
  } else {
    // Project [0,1,0] onto the plane perpendicular to shdRight.
    const t = dot([0, 1, 0] as V3, shdRight);
    spineUp = norm([
      0 - shdRight[0] * t,
      1 - shdRight[1] * t,
      0 - shdRight[2] * t,
    ]);
  }
  // Damp torso yaw so a head turn doesn't drag the chest around.
  return dampYaw(frameToQuat(shdRight, spineUp), TORSO_YAW_GAIN);
}
