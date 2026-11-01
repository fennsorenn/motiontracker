import { SignalNode, Quaternion, NormalizedPose } from '@vspark/shared/signal';
import type { VRMBoneName } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { valueIn, valueOut } from '@vspark/shared/node_decorators';
import {
  frameToQuat,
  quatToEulerXYZ,
  eulerXYZToQuat,
  torsoWorldQuat,
} from '../mocap_torso.js';

type Landmark = { x: number; y: number; z: number; visibility?: number };
type V3 = [number, number, number];

const BP = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
};

// MediaPipe Face Mesh canonical landmark indices used to build a stable head frame.
// The dense face mesh tracks head tilt/turn far more reliably than the coarse pose
// ear/nose points (which barely move when you tilt your head).
const FACE = {
  leftSide: 454, // subject's left cheek / tragion area
  rightSide: 234, // subject's right cheek / tragion area
  forehead: 10, // top-center of the forehead
  chin: 152, // bottom-center of the chin
};

const VIS = 0.5;
const ok = (lm: Landmark): boolean => (lm.visibility ?? 1) >= VIS;

function sub(a: Landmark, b: Landmark): V3 {
  return [a.x - b.x, a.y - b.y, a.z - b.z];
}
function lenV(v: V3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
function norm(v: V3): V3 {
  const l = lenV(v);
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

function qmul(a: Quaternion, b: Quaternion): Quaternion {
  return new Quaternion(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  );
}
function qinv(q: Quaternion): Quaternion {
  return new Quaternion(-q.x, -q.y, -q.z, q.w);
}

// Slerp from identity to q by factor t. Result q' satisfies q'^(1/t) = q (for small angles).
// Used to split a single body rotation across multiple spine bones so the bend distributes
// instead of concentrating in one joint.
function qSlerpFromIdentity(q: Quaternion, t: number): Quaternion {
  let w = q.w;
  // Take the shorter arc — flip sign if w < 0 (q and -q represent the same rotation).
  const sign = w < 0 ? -1 : 1;
  w *= sign;
  if (w > 0.9999) return new Quaternion(0, 0, 0, 1);
  const angle = Math.acos(w);
  const sinA = Math.sin(angle);
  const a = Math.sin((1 - t) * angle) / sinA; // weight on identity
  const b = Math.sin(t * angle) / sinA; // weight on q
  return new Quaternion(
    sign * b * q.x,
    sign * b * q.y,
    sign * b * q.z,
    a + sign * b * q.w
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// Coordinate conventions:
//
// MediaPipe poseWorldLandmarks:
//   Origin: hip midpoint. +Y=up, +Z=toward camera, +X=subject's right.
//
// VRM normalised pose T-pose (all bones at identity):
//   Spine/neck: +Y points up (toward crown), character faces -Z.
//
// For the spine: right = shoulder_right, up = spine_dir, forward derived.
// For the head:  right = ear-to-ear, up = derived from sagittal, forward derived.
//
// We use frameToQuat(right, up) which keeps rightTarget exact and orthogonalises up.
// ─────────────────────────────────────────────────────────────────────────────

// MediaPipe world landmarks: +Y down (image convention) and +Z away from camera. Flip both
// at the boundary so we work in the avatar's natural frame: +Y up, -Z = subject's front.
// Matches pose_arms_to_bones so arms and torso/head share one coordinate system.
function flipYZ(lm: Landmark): Landmark {
  return { x: lm.x, y: -lm.y, z: -lm.z, visibility: lm.visibility };
}

// Build a head-orientation frame from the dense face mesh, expressed in the avatar frame
// (+Y up, -Z forward). Returns {right, up} or null when the mesh is missing/degenerate.
// `right` follows the same convention as the pose path: subject-left side minus subject-right.
function faceHeadFrame(rawFace: Landmark[] | undefined): {
  right: V3;
  up: V3;
} | null {
  if (!rawFace || rawFace.length < 468) return null;
  const f = [
    rawFace[FACE.leftSide],
    rawFace[FACE.rightSide],
    rawFace[FACE.forehead],
    rawFace[FACE.chin],
  ];
  if (f.some((p) => !p)) return null;
  const right = norm(sub(flipYZ(f[0]), flipYZ(f[1])));
  const up = norm(sub(flipYZ(f[2]), flipYZ(f[3])));
  // Reject near-degenerate frames (right and up almost parallel → unstable basis).
  if (Math.abs(dot(right, up)) > 0.95) return null;
  return { right, up };
}



function convertPose(
  rawPts: Landmark[],
  calib: {
    pitchGain: number;
    yawGain: number;
    rollGain: number;
    restPitch: number;
  },
  faceRaw?: Landmark[]
): NormalizedPose {
  if (rawPts.length < 33) return new NormalizedPose();
  const pts = rawPts.map(flipYZ);

  const ls = pts[BP.leftShoulder],
    rs = pts[BP.rightShoulder];
  const lh = pts[BP.leftHip],
    rh = pts[BP.rightHip];
  const nose = pts[BP.nose];
  const lEar = pts[BP.leftEar],
    rEar = pts[BP.rightEar];

  const entries: [VRMBoneName, Quaternion][] = [];

  if (!ok(ls) || !ok(rs)) return new NormalizedPose();

  // In MediaPipe world landmarks, rightShoulder.x < leftShoulder.x (mirrored convention).
  // sub(ls, rs) gives the vector pointing in +X (subject's right).
  const shdRight = norm(sub(ls, rs));

  // ── Torso ─────────────────────────────────────────────────────────────────
  // Shared with pose_arms_to_bones (see mocap_torso.ts) so the arms hang off exactly this chest.
  // Its yaw is damped so a head turn doesn't drag the chest around (the neck, computed relative to
  // this torso below, absorbs the difference and the head still points the right way).
  const torsoQ = torsoWorldQuat(ls, rs, lh, rh);

  // Hips are intentionally left at identity so the legs and root position stay anchored.
  // The torso rotation is split across spine + chest as two local rotations whose product
  // is torsoQ, so the upper body bends through the spine rather than the hips.
  //
  //   chest_world = spine_local * chest_local = torsoQ
  //
  // We give each bone the same "half" rotation. Since both rotations are around the same axis,
  // half * half = full. That gives a natural distribution of the bend across the spine.
  const halfQ = qSlerpFromIdentity(torsoQ, 0.5);
  entries.push(['spine', halfQ]);
  entries.push(['chest', halfQ]);

  // Shoulder shrug is handled in pose_arms_to_bones, where it can be folded into the arm chain so
  // the clavicle lift doesn't drag the arms up with it.

  // ── Head ─────────────────────────────────────────────────────────────────
  {
    let headRight: V3;
    let headUp: V3;

    const faceFrame = faceHeadFrame(faceRaw);
    if (faceFrame) {
      // Preferred: dense face mesh gives stable tilt (roll), turn (yaw) and nod (pitch).
      // Any constant neutral offset is removed downstream by the head-neutral calibration.
      headRight = faceFrame.right;
      headUp = faceFrame.up;
    } else if (ok(lEar) && ok(rEar) && ok(nose)) {
      // In unified +Y-up, -Z-forward frame:
      //   leftEar  → +X side of head; rightEar → -X side.
      //   sub(lEar, rEar) points in +X (head right).
      headRight = norm(sub(lEar, rEar));

      // earMid → nose at neutral pose ≈ [0, -anatomicalDrop, -1]. The nose sits ~9° below
      // the horizontal ear plane, which we correct by rotating the sagittal-plane projection
      // back up around headRight.
      const earMid = mid(lEar, rEar);
      const rawToNose = norm(sub(nose, earMid));
      const dNose = dot(rawToNose, headRight);
      const sagNose: V3 = norm([
        rawToNose[0] - headRight[0] * dNose,
        rawToNose[1] - headRight[1] * dNose,
        rawToNose[2] - headRight[2] * dNose,
      ]);

      // Rodrigues rotation of sagNose around headRight by REST (radians). With sagNose pointing
      // down+forward at neutral gaze ([0, -y, +z], y,z > 0) and rotation axis = +X, a NEGATIVE
      // angle lifts the direction toward horizontal. Magnitude is anatomy-dependent; surfaced
      // as a calibration knob.
      const REST = calib.restPitch;
      const cosR = Math.cos(REST),
        sinR = Math.sin(REST);
      const hx = headRight[0],
        hy = headRight[1],
        hz = headRight[2];
      const sx = sagNose[0],
        sy = sagNose[1],
        sz = sagNose[2];
      const crs = cross(headRight, sagNose);
      const dp = dot(headRight, sagNose);
      const corrected: V3 = [
        sx * cosR + crs[0] * sinR + hx * dp * (1 - cosR),
        sy * cosR + crs[1] * sinR + hy * dp * (1 - cosR),
        sz * cosR + crs[2] * sinR + hz * dp * (1 - cosR),
      ];
      const correctedNorm = norm(corrected); // ≈ headForward at neutral (+Z in this frame)
      // headUp = cross(headForward, headRight). cross([0,0,1], [1,0,0]) = [0,1,0]. ✓
      headUp = norm(cross(correctedNorm, headRight));
    } else if (ok(nose)) {
      headRight = shdRight;
      const t = dot([0, 1, 0] as V3, headRight);
      headUp = norm([
        0 - headRight[0] * t,
        1 - headRight[1] * t,
        0 - headRight[2] * t,
      ]);
    } else {
      return new NormalizedPose(entries);
    }

    const worldHeadQ = frameToQuat(headRight, headUp);
    // Express relative to chest so it compounds correctly
    const localHeadQ = qmul(qinv(torsoQ), worldHeadQ);

    // Decompose to XYZ Euler in chest-local space, apply per-axis gain, recompose.
    // This lets us amplify each rotation axis independently to compensate MediaPipe's damping.
    const e = quatToEulerXYZ(localHeadQ);
    const calibratedHeadQ = eulerXYZToQuat(
      e.x * calib.pitchGain,
      e.y * calib.yawGain,
      e.z * calib.rollGain
    );

    // Only set neck — head is its child and inherits.
    entries.push(['neck', calibratedHeadQ]);
  }

  return new NormalizedPose(entries);
}

interface HeadCalibration {
  pitchGain: number; // multiplier on nod angle (X-axis rotation)
  yawGain: number; // multiplier on turn angle (Y)
  rollGain: number; // multiplier on tilt angle (Z)
  restPitch: number; // radians added to the nod axis to compensate anatomical neutral offset
}

@SignalNode({
  label: 'Pose → Torso/Head Bones',
  description:
    'Converts MediaPipe BlazePose 33-point world landmarks to VRM torso (spine+chest) + head (neck) local rotations. Hips are left at identity so legs stay anchored. Pitch/yaw/roll gain inputs amplify head rotation axes to compensate MediaPipe damping; `restPitch` shifts the nod neutral.',
  tags: ["mocap"],
  color: '#4a5a8a',
})
export class PoseTorsoHeadToBones extends Node {
  static readonly kind = 'pose_torso_head_to_bones';

  @valueIn('pose', 'LandmarkList') poseIn!: () => Landmark[] | undefined;
  // Optional dense face mesh — when present, the head frame is derived from it instead of the
  // coarse pose ear/nose points (better tilt/turn fidelity).
  @valueIn('face', 'LandmarkList') faceIn!: () => Landmark[] | undefined;
  @valueIn('enabled', 'Bool') enabledIn!: () => boolean | null | undefined;
  @valueIn('pitchGain', 'Float') pitchGain!: () => number | undefined;
  @valueIn('yawGain', 'Float') yawGain!: () => number | undefined;
  @valueIn('rollGain', 'Float') rollGain!: () => number | undefined;
  @valueIn('restPitch', 'Float') restPitch!: () => number | undefined;

  @valueOut('pose', 'NormalizedPose')
  poseOut = (): NormalizedPose | undefined => {
    const enabled = this.enabledIn() ?? true;
    if (!enabled) return new NormalizedPose();
    const pts = this.poseIn();
    if (!pts?.length) return undefined;
    const numIn = (v: unknown, d: number): number =>
      typeof v === 'number' && Number.isFinite(v) ? v : d;
    const calib: HeadCalibration = {
      pitchGain: numIn(this.pitchGain(), 2.0),
      yawGain: numIn(this.yawGain(), 1.0),
      rollGain: numIn(this.rollGain(), 1.0),
      restPitch: numIn(this.restPitch(), -0.43),
    };
    return convertPose(pts, calib, this.faceIn());
  };
}
