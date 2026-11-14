import type { GraphDescriptor } from '@vspark/shared/signal';

// VRM bone names whose neutral-pose bias is captured by the head_calib node.
// Limited to head/torso bones — arm and finger calibration would warrant separate nodes
// since arm corrections are sensitive to the IK target frame, not raw quaternion bias.
export const HEAD_CALIB_BONES = [
  'hips',
  'spine',
  'chest',
  'upperChest',
  'neck',
  'head',
  'jaw',
  'leftEye',
  'rightEye',
  // Clavicles — shrug is measured against an approximate neutral ratio, so the head-neutral
  // capture removes each performer's true resting shoulder offset.
  'leftShoulder',
  'rightShoulder',
  // Wrists — the hand-bone orientation carries a rest offset from noisy hand depth; capturing it
  // at neutral keeps the hands from sitting flicked-back when held in a relaxed pose.
  'leftHand',
  'rightHand',
] as const;

// pose_arms_to_bones solves each upper arm relative to the measured shoulder shrug, cancelling the
// clavicle lift so the arm still points at the elbow. head_calib then removes the shrug's neutral
// offset from the clavicle; without this the arm would be rotated by that offset too.
export const HEAD_CALIB_PRESERVE = [
  { bone: 'leftUpperArm', parents: ['leftShoulder'] },
  { bone: 'rightUpperArm', parents: ['rightShoulder'] },
] as const;

// VRM bone names captured by the finger_calib node. Held separate from head calibration
// because the user-facing capture ritual is different — fingers must be held in the avatar's
// T-pose finger rest (straight, thumb relaxed-out), not a body-neutral pose.
export const FINGER_CALIB_BONES = [
  'leftThumbMetacarpal',
  'leftThumbProximal',
  'leftThumbDistal',
  'leftIndexProximal',
  'leftIndexIntermediate',
  'leftIndexDistal',
  'leftMiddleProximal',
  'leftMiddleIntermediate',
  'leftMiddleDistal',
  'leftRingProximal',
  'leftRingIntermediate',
  'leftRingDistal',
  'leftLittleProximal',
  'leftLittleIntermediate',
  'leftLittleDistal',
  'rightThumbMetacarpal',
  'rightThumbProximal',
  'rightThumbDistal',
  'rightIndexProximal',
  'rightIndexIntermediate',
  'rightIndexDistal',
  'rightMiddleProximal',
  'rightMiddleIntermediate',
  'rightMiddleDistal',
  'rightRingProximal',
  'rightRingIntermediate',
  'rightRingDistal',
  'rightLittleProximal',
  'rightLittleIntermediate',
  'rightLittleDistal',
] as const;

// Left/right pairs used for symmetric mirror-fill during finger calibration —
// lets the user hold only one hand in rest while the other clicks the button.
export const FINGER_MIRROR_PAIRS = [
  ['leftThumbMetacarpal', 'rightThumbMetacarpal'],
  ['leftThumbProximal', 'rightThumbProximal'],
  ['leftThumbDistal', 'rightThumbDistal'],
  ['leftIndexProximal', 'rightIndexProximal'],
  ['leftIndexIntermediate', 'rightIndexIntermediate'],
  ['leftIndexDistal', 'rightIndexDistal'],
  ['leftMiddleProximal', 'rightMiddleProximal'],
  ['leftMiddleIntermediate', 'rightMiddleIntermediate'],
  ['leftMiddleDistal', 'rightMiddleDistal'],
  ['leftRingProximal', 'rightRingProximal'],
  ['leftRingIntermediate', 'rightRingIntermediate'],
  ['leftRingDistal', 'rightRingDistal'],
  ['leftLittleProximal', 'rightLittleProximal'],
  ['leftLittleIntermediate', 'rightLittleIntermediate'],
  ['leftLittleDistal', 'rightLittleDistal'],
] as const;

// Pipeline shape (per-stream):
//
//   mp_source (event: face/pose/leftHand/rightHand)
//     → unpack_* (UnpackEvent: splits event → trigger + stored value)
//       trigger → broadcast nodes
//       value (via pull) → converter node → pose_merge / bs_out
//
// Arm tracking has two modes — IK and quaternion — controlled by `useIk` in the component
// config. The mode is read by a `behavior_config` node and wired through to each arm-related
// node's `enabled` input. `useIk` defaults to false (quaternion arms).

// Helper to keep the descriptor readable: a behavior_config node that reads one field.
/**
 * Arm-solve technique switches: `armSolve.<field>` in the behavior config → a value port on
 * `pose_arms` (ArmSolveOptions) or `arm_smooth` (pose_smooth). Exposed so each technique's effect
 * can be compared live; the defaults are the full current solve.
 */
export const ARM_SOLVE_SWITCHES: readonly [
  string,
  'pose_arms' | 'arm_smooth',
  unknown,
][] = [
  ['mode', 'pose_arms', 'hinge'],
  ['keepLowVis', 'pose_arms', true],
  ['planeWristGate', 'pose_arms', true],
  ['handSwingOnly', 'pose_arms', true],
  ['swingFade', 'pose_arms', true],
  ['edgeOnConf', 'pose_arms', true],
  ['poseDorsalConf', 'pose_arms', true],
  ['smoothing', 'arm_smooth', true],
  ['jumpHold', 'arm_smooth', true],
];
/** Config fields whose target port is named differently. */
const ARM_SOLVE_PORT: Record<string, string> = { smoothing: 'enabled' };

function cfgNode(
  id: string,
  field: string,
  defaultValue: unknown,
  position: { x: number; y: number }
) {
  return {
    id,
    kind: 'behavior_config',
    position,
    defaultConfig: { field, defaultValue },
  };
}

export const MEDIAPIPE_PIPELINE_TEMPLATE: Omit<GraphDescriptor, 'id'> = {
  label: 'MediaPipe Tracker',
  readonly: true,
  nodes: [
    // ── Infrastructure ───────────────────────────────────────────────────────
    { id: 'scene_entity', kind: 'scene_entity', position: { x: 1400, y: 0 } },
    { id: 'comp_id', kind: 'behavior_id', position: { x: 1400, y: 80 } },
    // ── Entry point ──────────────────────────────────────────────────────────
    { id: 'mp_source', kind: 'mediapipe_source', position: { x: -120, y: 0 } },

    // ── Face stream ───────────────────────────────────────────────────────────
    // Face landmarks still drive the head frame (in pose_torso_head). Expressions
    // now come from MediaPipe's native ARKit blendshapes, mapped to VRM via the
    // same arkit_vrm_mapper trio as the VMC face input — fcl on by default;
    // expressions and passthrough are independently toggleable and summed so they
    // coexist.
    { id: 'unpack_face', kind: 'unpack_event', position: { x: 120, y: -160 } },
    { id: 'unpack_arkit', kind: 'unpack_event', position: { x: 120, y: -360 } },
    {
      id: 'arkit_fcl',
      kind: 'arkit_vrm_mapper',
      position: { x: 360, y: -460 },
      defaultConfig: { mode: 'fcl' },
    },
    {
      id: 'arkit_expr',
      kind: 'arkit_vrm_mapper',
      position: { x: 360, y: -360 },
      defaultConfig: { mode: 'expressions' },
    },
    {
      id: 'arkit_pass',
      kind: 'arkit_vrm_mapper',
      position: { x: 360, y: -260 },
      defaultConfig: { mode: 'passthrough' },
    },
    { id: 'bs_sum', kind: 'blendshapes_sum', position: { x: 560, y: -360 } },
    {
      id: 'bs_out',
      kind: 'blendshapes_broadcast',
      position: { x: 760, y: -360 },
    },
    // ARKit mapper config pairs — same field names as the VMC pipeline so the
    // same expression-mapping UI controls apply to both.
    cfgNode('cfg_fcl_en', 'nodeConfig.arkit_fcl_cfg.enabled', true, {
      x: -120,
      y: -560,
    }),
    cfgNode('cfg_fcl_map', 'nodeConfig.arkit_fcl_cfg.mapping', null, {
      x: -120,
      y: -520,
    }),
    cfgNode('cfg_expr_en', 'nodeConfig.arkit_expr_cfg.enabled', false, {
      x: -120,
      y: -480,
    }),
    cfgNode('cfg_expr_map', 'nodeConfig.arkit_expr_cfg.mapping', null, {
      x: -120,
      y: -440,
    }),
    cfgNode('cfg_pass_en', 'nodeConfig.arkit_pass_cfg.enabled', false, {
      x: -120,
      y: -400,
    }),
    cfgNode('cfg_pass_map', 'nodeConfig.arkit_pass_cfg.mapping', null, {
      x: -120,
      y: -360,
    }),

    // ── Pose stream ───────────────────────────────────────────────────────────
    { id: 'unpack_pose', kind: 'unpack_event', position: { x: 120, y: 0 } },
    {
      id: 'pose_torso_head',
      kind: 'pose_torso_head_to_bones',
      position: { x: 360, y: 0 },
    },
    // Quaternion-driven arms — used when IK is off
    {
      id: 'pose_arms',
      kind: 'pose_arms_to_bones',
      position: { x: 360, y: 80 },
    },
    // Eases arm bones whose landmarks MediaPipe can only estimate (low visibility, no hand)
    // instead of letting them jump.
    { id: 'arm_smooth', kind: 'pose_smooth', position: { x: 580, y: 80 } },

    // ── Left hand stream ──────────────────────────────────────────────────────
    { id: 'unpack_lh', kind: 'unpack_event', position: { x: 120, y: 200 } },
    {
      id: 'left_hand',
      kind: 'hand_landmarks_to_bones',
      position: { x: 360, y: 200 },
      defaultConfig: { side: 'left' },
    },

    // ── Right hand stream ─────────────────────────────────────────────────────
    { id: 'unpack_rh', kind: 'unpack_event', position: { x: 120, y: 320 } },
    {
      id: 'right_hand',
      kind: 'hand_landmarks_to_bones',
      position: { x: 360, y: 320 },
      defaultConfig: { side: 'right' },
    },

    // ── Merge all pose streams ────────────────────────────────────────────────
    { id: 'pose_merge', kind: 'pose_merge', position: { x: 800, y: 160 } },
    // ── Calibration: head/spine ──────────────────────────────────────────────
    // Removes neutral-pose bias for head + torso bones via capture/reset triggers.
    {
      id: 'head_calib',
      kind: 'body_calibration',
      position: { x: 1100, y: 160 },
    },
    {
      id: 'head_calib_capture',
      kind: 'component_trigger',
      position: { x: 1100, y: 60 },
      defaultConfig: { button: 'Capture head neutral' },
    },
    {
      id: 'head_calib_reset',
      kind: 'component_trigger',
      position: { x: 1260, y: 60 },
      defaultConfig: { button: 'Reset head' },
    },
    // ── Calibration: fingers ────────────────────────────────────────────────
    // Held separate from head — user holds one hand in the avatar's finger T-pose rest
    // (straight fingers, thumb relaxed-out). The higher hand at capture time wins; the
    // other side is mirrored from it.
    {
      id: 'finger_calib',
      kind: 'body_calibration',
      position: { x: 1250, y: 160 },
    },
    {
      id: 'finger_calib_capture',
      kind: 'component_trigger',
      position: { x: 1100, y: 260 },
      defaultConfig: { button: 'Capture finger neutral' },
    },
    {
      id: 'finger_calib_reset',
      kind: 'component_trigger',
      position: { x: 1260, y: 260 },
      defaultConfig: { button: 'Reset fingers' },
    },
    {
      id: 'hand_height',
      kind: 'hand_height_compare',
      position: { x: 1000, y: 260 },
    },
    { id: 'pose_out', kind: 'pose_broadcast', position: { x: 1400, y: 160 } },

    // ── IK targets stream ─────────────────────────────────────────────────────
    {
      id: 'ik_targets',
      kind: 'pose_ik_targets',
      position: { x: 800, y: 440 },
      defaultConfig: { smoothing: 0.25, referenceBone: 'chest' },
    },
    { id: 'ik_out', kind: 'ik_broadcast', position: { x: 1400, y: 440 } },

    // ── Config: useIk flag + NOT for inverting it ────────────────────────────
    cfgNode('cfg_useIk', 'useIk', false, { x: -120, y: 440 }),
    { id: 'not_useIk', kind: 'not_bool', position: { x: 120, y: 440 } },

    // ── Config: per-axis IK calibration ──────────────────────────────────────
    cfgNode('cfg_xScale', 'ikCalibration.xScale', 1, { x: 520, y: 500 }),
    cfgNode('cfg_yScale', 'ikCalibration.yScale', 1, { x: 520, y: 540 }),
    cfgNode('cfg_zScale', 'ikCalibration.zScale', 3, { x: 520, y: 580 }),
    cfgNode('cfg_xOffset', 'ikCalibration.xOffset', 0, { x: 520, y: 620 }),
    cfgNode('cfg_yOffset', 'ikCalibration.yOffset', 0, { x: 520, y: 660 }),
    cfgNode('cfg_zOffset', 'ikCalibration.zOffset', 0, { x: 520, y: 700 }),
    cfgNode('cfg_invertX', 'ikCalibration.invertX', false, { x: 520, y: 740 }),
    cfgNode('cfg_invertY', 'ikCalibration.invertY', false, { x: 520, y: 780 }),
    cfgNode('cfg_invertZ', 'ikCalibration.invertZ', false, { x: 520, y: 820 }),

    // ── Config: arm-solve technique switches (A/B comparison) ────────────────
    // See ArmSolveOptions in pose_arms_to_bones; defaults = the full current solve.
    ...ARM_SOLVE_SWITCHES.map(([field, , def], i) =>
      cfgNode(`cfg_arm_${field}`, `armSolve.${field}`, def, {
        x: 160,
        y: 680 + i * 40,
      })
    ),
    // ── Config: head calibration ─────────────────────────────────────────────
    cfgNode('cfg_head_pitchGain', 'headCalibration.pitchGain', 2.0, {
      x: 160,
      y: 500,
    }),
    cfgNode('cfg_head_yawGain', 'headCalibration.yawGain', 1.0, {
      x: 160,
      y: 540,
    }),
    cfgNode('cfg_head_rollGain', 'headCalibration.rollGain', 1.0, {
      x: 160,
      y: 580,
    }),
    cfgNode('cfg_head_restPitch', 'headCalibration.restPitch', -0.43, {
      x: 160,
      y: 620,
    }),
  ],
  edges: [
    // ── mp_source → unpackers (event) ─────────────────────────────────────────
    {
      fromNodeId: 'mp_source',
      fromPort: 'face',
      toNodeId: 'unpack_face',
      toPort: 'event',
    },
    {
      fromNodeId: 'mp_source',
      fromPort: 'pose',
      toNodeId: 'unpack_pose',
      toPort: 'event',
    },
    {
      fromNodeId: 'mp_source',
      fromPort: 'leftHand',
      toNodeId: 'unpack_lh',
      toPort: 'event',
    },
    {
      fromNodeId: 'mp_source',
      fromPort: 'rightHand',
      toNodeId: 'unpack_rh',
      toPort: 'event',
    },

    // ── Face mesh → head frame (tilt/turn) in the torso/head node ─────────────
    {
      fromNodeId: 'unpack_face',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'face',
      kind: 'value',
    },

    // ── ARKit blendshapes: source → unpack → mapper trio → sum → broadcast ────
    {
      fromNodeId: 'mp_source',
      fromPort: 'arkit',
      toNodeId: 'unpack_arkit',
      toPort: 'event',
    },
    {
      fromNodeId: 'unpack_arkit',
      fromPort: 'trigger',
      toNodeId: 'bs_out',
      toPort: 'trigger',
    },
    {
      fromNodeId: 'unpack_arkit',
      fromPort: 'value',
      toNodeId: 'arkit_fcl',
      toPort: 'arkit',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_arkit',
      fromPort: 'value',
      toNodeId: 'arkit_expr',
      toPort: 'arkit',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_arkit',
      fromPort: 'value',
      toNodeId: 'arkit_pass',
      toPort: 'arkit',
      kind: 'value',
    },
    {
      fromNodeId: 'arkit_fcl',
      fromPort: 'blendshapes',
      toNodeId: 'bs_sum',
      toPort: 'sources',
      kind: 'list',
    },
    {
      fromNodeId: 'arkit_expr',
      fromPort: 'blendshapes',
      toNodeId: 'bs_sum',
      toPort: 'sources',
      kind: 'list',
    },
    {
      fromNodeId: 'arkit_pass',
      fromPort: 'blendshapes',
      toNodeId: 'bs_sum',
      toPort: 'sources',
      kind: 'list',
    },
    {
      fromNodeId: 'bs_sum',
      fromPort: 'blendshapes',
      toNodeId: 'bs_out',
      toPort: 'blendshapes',
      kind: 'value',
    },
    // Mapper enable/mapping config.
    {
      fromNodeId: 'cfg_fcl_en',
      fromPort: 'value',
      toNodeId: 'arkit_fcl',
      toPort: 'enabled',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_fcl_map',
      fromPort: 'value',
      toNodeId: 'arkit_fcl',
      toPort: 'mapping',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_expr_en',
      fromPort: 'value',
      toNodeId: 'arkit_expr',
      toPort: 'enabled',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_expr_map',
      fromPort: 'value',
      toNodeId: 'arkit_expr',
      toPort: 'mapping',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_pass_en',
      fromPort: 'value',
      toNodeId: 'arkit_pass',
      toPort: 'enabled',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_pass_map',
      fromPort: 'value',
      toNodeId: 'arkit_pass',
      toPort: 'mapping',
      kind: 'value',
    },
    {
      fromNodeId: 'scene_entity',
      fromPort: 'nodeId',
      toNodeId: 'bs_out',
      toPort: 'nodeId',
      kind: 'value',
    },
    {
      fromNodeId: 'comp_id',
      fromPort: 'id',
      toNodeId: 'bs_out',
      toPort: 'behaviorId',
      kind: 'value',
    },

    // ── Pose: trigger → pose_out; value pulled by torso/head + arms ──────────
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'trigger',
      toNodeId: 'pose_out',
      toPort: 'trigger',
    },
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'pose',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'value',
      toNodeId: 'pose_arms',
      toPort: 'pose',
      kind: 'value',
    },

    // ── Hands: unpacked values pulled by converters ───────────────────────────
    {
      fromNodeId: 'unpack_lh',
      fromPort: 'value',
      toNodeId: 'left_hand',
      toPort: 'landmarks',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_rh',
      fromPort: 'value',
      toNodeId: 'right_hand',
      toPort: 'landmarks',
      kind: 'value',
    },
    // Hand landmarks also feed the quaternion-arm node so it can set wrist orientation.
    {
      fromNodeId: 'unpack_lh',
      fromPort: 'value',
      toNodeId: 'pose_arms',
      toPort: 'leftHand',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_rh',
      fromPort: 'value',
      toNodeId: 'pose_arms',
      toPort: 'rightHand',
      kind: 'value',
    },

    // ── Merge: torso/head + arms (quat) + both hands → pose_merge ────────────
    // pose_arms returns empty pose when disabled, so it's safe to wire unconditionally.
    {
      fromNodeId: 'pose_torso_head',
      fromPort: 'pose',
      toNodeId: 'pose_merge',
      toPort: 'poses',
      kind: 'list',
    },
    {
      fromNodeId: 'pose_arms',
      fromPort: 'pose',
      toNodeId: 'arm_smooth',
      toPort: 'pose',
    },
    {
      fromNodeId: 'pose_arms',
      fromPort: 'confidence',
      toNodeId: 'arm_smooth',
      toPort: 'confidence',
    },
    {
      fromNodeId: 'arm_smooth',
      fromPort: 'pose',
      toNodeId: 'pose_merge',
      toPort: 'poses',
      kind: 'list',
    },
    {
      fromNodeId: 'left_hand',
      fromPort: 'pose',
      toNodeId: 'pose_merge',
      toPort: 'poses',
      kind: 'list',
    },
    {
      fromNodeId: 'right_hand',
      fromPort: 'pose',
      toNodeId: 'pose_merge',
      toPort: 'poses',
      kind: 'list',
    },
    {
      fromNodeId: 'pose_merge',
      fromPort: 'pose',
      toNodeId: 'head_calib',
      toPort: 'pose',
      kind: 'value',
    },
    {
      fromNodeId: 'head_calib',
      fromPort: 'pose',
      toNodeId: 'finger_calib',
      toPort: 'pose',
      kind: 'value',
    },
    {
      fromNodeId: 'finger_calib',
      fromPort: 'pose',
      toNodeId: 'pose_out',
      toPort: 'pose',
      kind: 'value',
    },
    // Manual triggers → body_calibration capture/reset event ports.
    {
      fromNodeId: 'head_calib_capture',
      fromPort: 'trigger',
      toNodeId: 'head_calib',
      toPort: 'capture',
    },
    {
      fromNodeId: 'head_calib_reset',
      fromPort: 'trigger',
      toNodeId: 'head_calib',
      toPort: 'reset',
    },
    {
      fromNodeId: 'finger_calib_capture',
      fromPort: 'trigger',
      toNodeId: 'finger_calib',
      toPort: 'capture',
    },
    {
      fromNodeId: 'finger_calib_reset',
      fromPort: 'trigger',
      toNodeId: 'finger_calib',
      toPort: 'reset',
    },
    // Hand-height comparator feeds the mirror-source decision: higher hand wins.
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'value',
      toNodeId: 'hand_height',
      toPort: 'pose',
      kind: 'value',
    },
    {
      fromNodeId: 'hand_height',
      fromPort: 'side',
      toNodeId: 'finger_calib',
      toPort: 'mirrorSource',
      kind: 'value',
    },
    {
      fromNodeId: 'scene_entity',
      fromPort: 'nodeId',
      toNodeId: 'pose_out',
      toPort: 'nodeId',
      kind: 'value',
    },
    {
      fromNodeId: 'comp_id',
      fromPort: 'id',
      toNodeId: 'pose_out',
      toPort: 'behaviorId',
      kind: 'value',
    },

    // ── IK targets: triggered by pose; pulls pose + hand landmark lists ───────
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'trigger',
      toNodeId: 'ik_out',
      toPort: 'trigger',
    },
    {
      fromNodeId: 'unpack_pose',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'pose',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_lh',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'leftHand',
      kind: 'value',
    },
    {
      fromNodeId: 'unpack_rh',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'rightHand',
      kind: 'value',
    },
    {
      fromNodeId: 'ik_targets',
      fromPort: 'targets',
      toNodeId: 'ik_out',
      toPort: 'targets',
      kind: 'value',
    },
    {
      fromNodeId: 'scene_entity',
      fromPort: 'nodeId',
      toNodeId: 'ik_out',
      toPort: 'nodeId',
      kind: 'value',
    },

    // ── Arm-solve technique switches ────────────────────────────────────────
    ...ARM_SOLVE_SWITCHES.map(([field, toNodeId]) => ({
      fromNodeId: `cfg_arm_${field}`,
      fromPort: 'value',
      toNodeId,
      toPort: ARM_SOLVE_PORT[field] ?? field,
      kind: 'value' as const,
    })),
    // ── Enable gates ──────────────────────────────────────────────────────────
    // useIk → ik_targets.enabled, ik_out.enabled
    {
      fromNodeId: 'cfg_useIk',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'enabled',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_useIk',
      fromPort: 'value',
      toNodeId: 'ik_out',
      toPort: 'enabled',
      kind: 'value',
    },
    // useIk → NOT → pose_arms.enabled (arms-by-quat is the inverse)
    {
      fromNodeId: 'cfg_useIk',
      fromPort: 'value',
      toNodeId: 'not_useIk',
      toPort: 'value',
      kind: 'value',
    },
    {
      fromNodeId: 'not_useIk',
      fromPort: 'result',
      toNodeId: 'pose_arms',
      toPort: 'enabled',
      kind: 'value',
    },

    // ── IK calibration: config nodes → ik_targets value ports ────────────────
    {
      fromNodeId: 'cfg_xScale',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'xScale',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_yScale',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'yScale',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_zScale',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'zScale',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_xOffset',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'xOffset',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_yOffset',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'yOffset',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_zOffset',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'zOffset',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_invertX',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'invertX',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_invertY',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'invertY',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_invertZ',
      fromPort: 'value',
      toNodeId: 'ik_targets',
      toPort: 'invertZ',
      kind: 'value',
    },

    // ── Head calibration: config nodes → pose_torso_head value ports ─────────
    {
      fromNodeId: 'cfg_head_pitchGain',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'pitchGain',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_head_yawGain',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'yawGain',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_head_rollGain',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'rollGain',
      kind: 'value',
    },
    {
      fromNodeId: 'cfg_head_restPitch',
      fromPort: 'value',
      toNodeId: 'pose_torso_head',
      toPort: 'restPitch',
      kind: 'value',
    },
  ],
};

export function makeMediapipeGraphDescriptor(
  behaviorId: string
): GraphDescriptor {
  return {
    ...MEDIAPIPE_PIPELINE_TEMPLATE,
    id: `mediapipe_tracker:${behaviorId}`,
  };
}
