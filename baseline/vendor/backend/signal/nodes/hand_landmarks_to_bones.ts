import { SignalNode, Quaternion, NormalizedPose } from '@vspark/shared/signal';
import type { VRMBoneName } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { valueIn, valueOut } from '@vspark/shared/node_decorators';

type Landmark = { x: number; y: number; z: number; visibility?: number };
type V3 = [number, number, number];

// MediaPipe Hand 21-point indices.
// https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
const H = {
  wrist: 0,
  thumbCmc: 1,
  thumbMcp: 2,
  thumbIp: 3,
  thumbTip: 4,
  indexMcp: 5,
  indexPip: 6,
  indexDip: 7,
  indexTip: 8,
  middleMcp: 9,
  middlePip: 10,
  middleDip: 11,
  middleTip: 12,
  ringMcp: 13,
  ringPip: 14,
  ringDip: 15,
  ringTip: 16,
  littleMcp: 17,
  littlePip: 18,
  littleDip: 19,
  littleTip: 20,
};

function sub(a: Landmark, b: Landmark): V3 {
  return [a.x - b.x, a.y - b.y, a.z - b.z];
}
function scale(v: V3, s: number): V3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}
function len(v: V3): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
function norm(v: V3): V3 {
  const l = len(v);
  return l < 1e-9 ? [0, 0, 1] : [v[0] / l, v[1] / l, v[2] / l];
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

// Minimal-arc rotation from unit vector `from` to unit vector `to`.
function rotFromTo(from: V3, to: V3): Quaternion {
  const d = dot(from, to);
  if (d > 0.9999) return Quaternion.IDENTITY;
  if (d < -0.9999) {
    const perp: V3 = Math.abs(from[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const ax = norm(cross(from, perp));
    return new Quaternion(ax[0], ax[1], ax[2], 0);
  }
  const ax = norm(cross(from, to));
  const s = Math.sin(Math.acos(d) / 2);
  return new Quaternion(
    ax[0] * s,
    ax[1] * s,
    ax[2] * s,
    Math.cos(Math.acos(d) / 2)
  );
}

// Rotate vector v by quaternion q.
function qvec(q: Quaternion, v: V3): V3 {
  const ix = q.w * v[0] + q.y * v[2] - q.z * v[1];
  const iy = q.w * v[1] + q.z * v[0] - q.x * v[2];
  const iz = q.w * v[2] + q.x * v[1] - q.y * v[0];
  const iw = -q.x * v[0] - q.y * v[1] - q.z * v[2];
  return [
    ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
    iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
    iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
  ];
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

// Quaternion from an axis (unit) and angle (radians).
function axisAngle(axis: V3, angle: number): Quaternion {
  const s = Math.sin(angle / 2);
  return new Quaternion(
    axis[0] * s,
    axis[1] * s,
    axis[2] * s,
    Math.cos(angle / 2)
  );
}

// Anatomical flexion range for a single finger segment: a little hyperextension, up to a
// strong curl. Keeps the joint from ever reaching the nonsensical 180° fold.
const FLEX_MIN = -0.2; // ~ -11°
const FLEX_MAX = 1.9; //  ~ 109°
// Abduction/adduction (finger fan-out) range at the knuckle. Smaller than flexion — fingers only
// splay a modest amount.
const SPREAD_MAX = 0.45; // ~ 26°

// Thumb. Its base sits splayed toward the thumb side and it curls across the palm toward the little
// finger rather than straight down, so it gets its own rest direction and hinge. In the hand-local
// frame (both sides, checked on real Holistic output) the palm faces -Y and the thumb side is +Z.
const THUMB_SPLAY = 0.6; // ~35°: rest metacarpal direction, from the finger axis toward the thumb
const THUMB_CMC_MAX = 1.2; // ~70°: largest metacarpal deviation from that rest
const THUMB_FLEX_MIN = -0.35; // ~ -20°: thumb joints hyperextend further than fingers
const THUMB_FLEX_MAX = 1.6; // ~ 92°
// Thumb curl direction: half toward the palm, mostly toward the little-finger side (-Z).
const THUMB_CURL: V3 = norm([0, -0.5, -1]);

// ─────────────────────────────────────────────────────────────────────────────
// Hand coordinate frame
//
// MediaPipe hand landmarks are in image-normalised space:
//   x: 0=left edge, 1=right edge (of bounding box)
//   y: 0=top, 1=bottom (inverted Y vs 3D)
//   z: depth relative to wrist (negative = further from camera)
//
// We build a hand-local 3D frame:
//   fingerAxis  = wrist → middle_mcp (along hand, primary axis)
//   palmNormal  = cross(index_mcp−wrist, pinky_mcp−wrist), negated for the right hand. Despite
//                 the name this is the BACK-of-hand (dorsal) normal for both hands — checked on
//                 real Holistic output for hands whose facing is unambiguous.
//   thumbAxis   = cross(fingerAxis, palmNormal)
//
// In VRM T-pose, for the LEFT hand:
//   The hand bone's +Y axis points from wrist toward middle fingertip.
//   The back of the hand faces +Y (palm down) and the thumb side is +Z (forward).
//
// For the RIGHT hand the fingers point along -X; back of hand +Y and thumb side +Z as on the left.
//
// Strategy:
//   1. Build the hand frame from landmarks.
//   2. Express the VRM T-pose hand axes in MediaPipe hand-local coords.
//   3. wristQ = rotation from VRM T-pose hand frame → observed hand frame.
//      This is the normalised-pose local rotation for the hand bone.
//   4. For each finger segment, compute the rotation of the segment direction
//      relative to the parent segment direction, expressed in hand-local space.
//      This gives the local rotation for each finger joint.
// ─────────────────────────────────────────────────────────────────────────────

function buildHandFrame(
  pts: Landmark[],
  side: 'left' | 'right'
): {
  fingerAxis: V3; // wrist → middle_mcp
  palmNormal: V3; // out of palm (toward viewer for right palm-facing-camera)
  thumbAxis: V3; // toward thumb side
  handToWorld: Quaternion; // rotation from VRM T-pose hand frame → MediaPipe image frame
} | null {
  const wrist = pts[H.wrist];
  const iMcp = pts[H.indexMcp];
  const mMcp = pts[H.middleMcp];
  const pMcp = pts[H.littleMcp];

  const fingerAxis = norm(sub(mMcp, wrist));
  // Build palm normal from index and pinky MCP relative to wrist
  const toIndex = sub(iMcp, wrist);
  const toPinky = sub(pMcp, wrist);
  let palmNormal = norm(cross(toIndex, toPinky));

  // For right hand, flip palm normal (mirrored image-space convention)
  if (side === 'right') palmNormal = scale(palmNormal, -1);

  // Ensure palmNormal is perpendicular to fingerAxis
  const projected = scale(fingerAxis, dot(palmNormal, fingerAxis));
  palmNormal = norm([
    palmNormal[0] - projected[0],
    palmNormal[1] - projected[1],
    palmNormal[2] - projected[2],
  ]);

  const thumbAxis = norm(cross(fingerAxis, palmNormal));

  // VRM T-pose hand frame (matches the upper-arm convention):
  //   Left hand:  fingerAxis (wrist→fingers) = +X,  dorsal (back of hand) ≈ +Y, thumb side = +Z
  //   Right hand: fingerAxis = -X,                  dorsal ≈ +Y,                thumb side = +Z
  // We only need two axes to constrain the rotation; we use fingerAxis + dorsal.
  const vrmFingerAxis: V3 = side === 'left' ? [1, 0, 0] : [-1, 0, 0];
  const vrmDorsal: V3 = [0, 1, 0]; // back-of-hand points up at T-pose

  // Step 1: swing VRM fingerAxis → observed fingerAxis
  const swingQ = rotFromTo(vrmFingerAxis, fingerAxis);

  // Step 2: twist around fingerAxis so VRM dorsal maps to observed palmNormal
  const dorsalAfterSwing = qvec(swingQ, vrmDorsal);
  const pa = dot(dorsalAfterSwing, fingerAxis);
  const dorsalProj = norm([
    dorsalAfterSwing[0] - fingerAxis[0] * pa,
    dorsalAfterSwing[1] - fingerAxis[1] * pa,
    dorsalAfterSwing[2] - fingerAxis[2] * pa,
  ]);
  const pb = dot(palmNormal, fingerAxis);
  const nProj = norm([
    palmNormal[0] - fingerAxis[0] * pb,
    palmNormal[1] - fingerAxis[1] * pb,
    palmNormal[2] - fingerAxis[2] * pb,
  ]);

  const twistAngle = Math.atan2(
    dot(cross(dorsalProj, nProj), fingerAxis),
    dot(dorsalProj, nProj)
  );
  const twistQ = new Quaternion(
    fingerAxis[0] * Math.sin(twistAngle / 2),
    fingerAxis[1] * Math.sin(twistAngle / 2),
    fingerAxis[2] * Math.sin(twistAngle / 2),
    Math.cos(twistAngle / 2)
  );
  const handToWorld = qmul(twistQ, swingQ);

  return { fingerAxis, palmNormal, thumbAxis, handToWorld };
}

// For a finger segment from→to, compute the local rotation relative to the
// parent quaternion (accumulated from wrist outward).
//
// VRM finger bones at T-pose rest extend along ±X (same axis as the upper arm), and the palm
// faces -Y (dorsal/back-of-hand is +Y). Finger joints are hinges: they flex by curling the rest
// direction toward the palm about a single axis. We model exactly that — measure the flexion
// angle in the rest→palm plane and rotate about the fixed hinge axis, clamped to an anatomical
// range. This avoids the rotFromTo degeneracy that made segments snap between 0° and 180° when
// MediaPipe's unreliable hand depth pushed the observed direction near anti-parallel to rest.
//
// For the middle and tip joints (no spread) the size of the bend comes from the angle between this
// segment and the previous one (`prevIdx`→`fromIdx`), as Kalidokit does. That angle doesn't depend
// on the hand frame or on the clamped rotations further up the finger, so errors there no longer
// compound toward the fingertip; the frame only decides curl vs hyperextension.
function fingerSegmentLocal(
  pts: Landmark[],
  fromIdx: number,
  toIdx: number,
  parentWorldQ: Quaternion,
  restDir: V3,
  prevIdx: number,
  allowSpread = false
): Quaternion {
  const dir = norm(sub(pts[toIdx], pts[fromIdx]));
  // Express the observed direction in parent-local space.
  const localDir = qvec(qinv(parentWorldQ), dir);
  const palmLocal: V3 = [0, -1, 0]; // palm direction in VRM hand-local rest frame
  // Signed flexion angle: 0 when aligned with rest, positive as it curls toward the palm.
  let flex = Math.atan2(dot(localDir, palmLocal), dot(localDir, restDir));
  if (!allowSpread) flex = signedBend(pts, prevIdx, fromIdx, toIdx, flex, FLEX_MIN);
  const flexClamped = Math.max(FLEX_MIN, Math.min(FLEX_MAX, flex));
  // Hinge axis = rest × palm; rotating restDir about it by +flex sweeps toward the palm.
  const hinge = norm(cross(restDir, palmLocal));
  const qFlex = axisAngle(hinge, flexClamped);
  if (!allowSpread) return qFlex;
  // Spread (abduction): the sideways angle of the finger within the palm plane, about the palm
  // normal. Measured from the finger direction with its palm-normal component removed, so it's
  // independent of how far the finger is curled.
  const sideAxis = norm(cross(palmLocal, restDir)); // in-palm sideways direction
  const planar: V3 = [localDir[0], 0, localDir[2]]; // drop palm-normal (Y) component
  const pl = len(planar);
  if (pl < 0.2) return qFlex; // too curled to read spread reliably
  const planarN: V3 = [planar[0] / pl, planar[1] / pl, planar[2] / pl];
  const spread = Math.atan2(dot(planarN, sideAxis), dot(planarN, restDir));
  const spreadClamped = Math.max(-SPREAD_MAX, Math.min(SPREAD_MAX, spread));
  // +spread rotates restDir toward sideAxis about the palm normal [0,-1,0].
  const qSpread = axisAngle([0, -1, 0], spreadClamped);
  return qmul(qSpread, qFlex);
}

// Unsigned angle at `b` between segments a→b and b→c, signed by the frame-derived flex estimate.
// A backward reading is only believed while it stays within the hyperextension limit — anything
// larger is a curl whose direction the (noisier) frame got wrong.
function signedBend(
  pts: Landmark[],
  a: number,
  b: number,
  c: number,
  frameFlex: number,
  hyperMin: number
): number {
  const u = norm(sub(pts[b], pts[a]));
  const v = norm(sub(pts[c], pts[b]));
  const bend = Math.acos(Math.max(-1, Math.min(1, dot(u, v))));
  return frameFlex < 0 && bend <= -hyperMin ? -bend : bend;
}

// Thumb metacarpal (saddle joint, two degrees of freedom): the minimal rotation from the thumb's
// rest direction to the observed CMC→MCP direction, capped at THUMB_CMC_MAX.
function thumbMetacarpalLocal(
  pts: Landmark[],
  handToWorld: Quaternion,
  thumbRest: V3
): Quaternion {
  const localDir = qvec(
    qinv(handToWorld),
    norm(sub(pts[H.thumbMcp], pts[H.thumbCmc]))
  );
  const angle = Math.acos(Math.max(-1, Math.min(1, dot(thumbRest, localDir))));
  if (angle < 1e-6) return Quaternion.IDENTITY;
  const axis = norm(cross(thumbRest, localDir));
  return axisAngle(axis, Math.min(angle, THUMB_CMC_MAX));
}

// Thumb MCP / IP: hinges about the thumb's own axis (rest × curl), sized like the finger joints.
function thumbHingeLocal(
  pts: Landmark[],
  prevIdx: number,
  fromIdx: number,
  toIdx: number,
  parentWorldQ: Quaternion,
  thumbRest: V3
): Quaternion {
  const hinge = norm(cross(thumbRest, THUMB_CURL));
  const curl = cross(hinge, thumbRest); // THUMB_CURL made perpendicular to the rest direction
  const localDir = qvec(
    qinv(parentWorldQ),
    norm(sub(pts[toIdx], pts[fromIdx]))
  );
  const frameFlex = Math.atan2(dot(localDir, curl), dot(localDir, thumbRest));
  const flex = signedBend(pts, prevIdx, fromIdx, toIdx, frameFlex, THUMB_FLEX_MIN);
  return axisAngle(hinge, Math.max(THUMB_FLEX_MIN, Math.min(THUMB_FLEX_MAX, flex)));
}

function convertHand(pts: Landmark[], side: 'left' | 'right'): NormalizedPose {
  if (pts.length < 21) return new NormalizedPose();

  const frame = buildHandFrame(pts, side);
  if (!frame) return new NormalizedPose();

  const { handToWorld } = frame;
  const entries: [VRMBoneName, Quaternion][] = [];
  const L = side === 'left';
  // Finger bones extend along ±X from the hand at rest (same convention as upper arm).
  const restDir: V3 = L ? [1, 0, 0] : [-1, 0, 0];

  // Hand bone wrist rotation — this is the wrist local rotation in VRM normalised pose
  // (relative to the parent lowerArm bind pose, which in T-pose = identity).
  // We intentionally omit it here because the arm IK sets the hand bone orientation.
  // Only finger joints below the hand bone are set here.

  // Accumulated world rotation for parent tracking (starts from handToWorld)
  // For each finger: MCP is child of hand, PIP child of MCP, DIP child of PIP.
  // Parent of MCP = hand bone = handToWorld.

  const fingers: Array<{
    mcp: number;
    pip: number;
    dip: number;
    tip: number;
    mcpBone: VRMBoneName;
    pipBone: VRMBoneName;
    dipBone: VRMBoneName;
  }> = [
    {
      mcp: H.indexMcp,
      pip: H.indexPip,
      dip: H.indexDip,
      tip: H.indexTip,
      mcpBone: L ? 'leftIndexProximal' : 'rightIndexProximal',
      pipBone: L ? 'leftIndexIntermediate' : 'rightIndexIntermediate',
      dipBone: L ? 'leftIndexDistal' : 'rightIndexDistal',
    },
    {
      mcp: H.middleMcp,
      pip: H.middlePip,
      dip: H.middleDip,
      tip: H.middleTip,
      mcpBone: L ? 'leftMiddleProximal' : 'rightMiddleProximal',
      pipBone: L ? 'leftMiddleIntermediate' : 'rightMiddleIntermediate',
      dipBone: L ? 'leftMiddleDistal' : 'rightMiddleDistal',
    },
    {
      mcp: H.ringMcp,
      pip: H.ringPip,
      dip: H.ringDip,
      tip: H.ringTip,
      mcpBone: L ? 'leftRingProximal' : 'rightRingProximal',
      pipBone: L ? 'leftRingIntermediate' : 'rightRingIntermediate',
      dipBone: L ? 'leftRingDistal' : 'rightRingDistal',
    },
    {
      mcp: H.littleMcp,
      pip: H.littlePip,
      dip: H.littleDip,
      tip: H.littleTip,
      mcpBone: L ? 'leftLittleProximal' : 'rightLittleProximal',
      pipBone: L ? 'leftLittleIntermediate' : 'rightLittleIntermediate',
      dipBone: L ? 'leftLittleDistal' : 'rightLittleDistal',
    },
  ];

  for (const f of fingers) {
    // MCP local (relative to hand = handToWorld). Knuckles carry the spread DOF.
    const mcpLocal = fingerSegmentLocal(
      pts,
      f.mcp,
      f.pip,
      handToWorld,
      restDir,
      H.wrist,
      true
    );
    entries.push([f.mcpBone, mcpLocal]);

    // PIP local (relative to MCP world = handToWorld * mcpLocal)
    const mcpWorld = qmul(handToWorld, mcpLocal);
    const pipLocal = fingerSegmentLocal(
      pts,
      f.pip,
      f.dip,
      mcpWorld,
      restDir,
      f.mcp
    );
    entries.push([f.pipBone, pipLocal]);

    // DIP local (relative to PIP world = mcpWorld * pipLocal)
    const pipWorld = qmul(mcpWorld, pipLocal);
    const dipLocal = fingerSegmentLocal(
      pts,
      f.dip,
      f.tip,
      pipWorld,
      restDir,
      f.pip
    );
    entries.push([f.dipBone, dipLocal]);
  }

  // Thumb (CMC→MCP→IP→Tip): own rest direction and hinge — see THUMB_SPLAY / THUMB_CURL.
  const thumbRest = norm([
    restDir[0] * Math.cos(THUMB_SPLAY),
    0,
    Math.sin(THUMB_SPLAY),
  ]);
  const thumbCmcLocal = thumbMetacarpalLocal(pts, handToWorld, thumbRest);
  entries.push([
    L ? 'leftThumbMetacarpal' : 'rightThumbMetacarpal',
    thumbCmcLocal,
  ]);
  const thumbCmcWorld = qmul(handToWorld, thumbCmcLocal);

  const thumbMcpLocal = thumbHingeLocal(
    pts,
    H.thumbCmc,
    H.thumbMcp,
    H.thumbIp,
    thumbCmcWorld,
    thumbRest
  );
  entries.push([L ? 'leftThumbProximal' : 'rightThumbProximal', thumbMcpLocal]);
  const thumbMcpWorld = qmul(thumbCmcWorld, thumbMcpLocal);

  const thumbIpLocal = thumbHingeLocal(
    pts,
    H.thumbMcp,
    H.thumbIp,
    H.thumbTip,
    thumbMcpWorld,
    thumbRest
  );
  entries.push([L ? 'leftThumbDistal' : 'rightThumbDistal', thumbIpLocal]);

  return new NormalizedPose(entries);
}

@SignalNode({
  label: 'Hand Landmarks → Bones',
  description:
    'Converts MediaPipe 21-point image-space hand landmarks to VRM finger joint quaternions using a hand-local coordinate frame.',
  tags: ["mocap"],
  color: '#4a5a8a',
})
export class HandLandmarksToBones extends Node {
  static readonly kind = 'hand_landmarks_to_bones';

  @valueIn('landmarks', 'LandmarkList') landmarks!: () => Landmark[] | undefined;
  @valueIn('side', 'String') side!: () => string | undefined;

  @valueOut('pose', 'NormalizedPose')
  pose = (): NormalizedPose | undefined => {
    const pts = this.landmarks();
    if (!pts?.length) return undefined;
    const side = this.side() ?? 'left';
    return convertHand(pts, side as 'left' | 'right');
  };
}
