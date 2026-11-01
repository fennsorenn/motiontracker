/**
 * Standalone driver for the vendored vspark MediaPipe → VRM solver.
 *
 * Reproduces the pose branch of vspark's `MEDIAPIPE_PIPELINE_TEMPLATE` (quaternion-arm mode, the
 * default) without vspark's signal-graph engine:
 *
 *   pose ──────┬─ pose_torso_head_to_bones ─────────────────┐
 *   face ──────┘                                           │
 *   pose, hands ─ pose_arms_to_bones ─ pose_smooth ────────┼─ pose_merge ─ head_calib ─ finger_calib
 *   leftHand ──── hand_landmarks_to_bones (left) ──────────┤                              ↑ mirrorSource
 *   rightHand ─── hand_landmarks_to_bones (right) ─────────┘          pose ─ hand_height_compare
 *
 * then, optionally, the frontend stage from vspark's Viewport: a per-bone One Euro filter run once
 * per render frame on the latest pose (and the second-order dynamics filter, off by default).
 *
 * Fidelity notes (see SOURCE.md):
 *  - vspark only fires a landmark stream when present, and `unpack_event` keeps the last payload,
 *    so a hand or face that drops out keeps feeding its LAST value. `holdMissing: true` (default)
 *    reproduces that; `false` feeds `undefined` instead.
 *  - Output is produced only on frames that carry pose landmarks (the pose stream triggers the
 *    broadcast), exactly like vspark.
 *  - `Date.now()` is pinned to the frame timestamp while the solver runs (pose_smooth reads it).
 */
import * as THREE from 'three';
import type { NormalizedPose } from '@vspark/shared/signal';
import type { Node, NodeBindContext, Thunk, Emitter } from '@vspark/shared/node';
import { PoseTorsoHeadToBones } from '../vendor/backend/signal/nodes/pose_torso_head_to_bones.js';
import { PoseArmsToBones } from '../vendor/backend/signal/nodes/pose_arms_to_bones.js';
import { PoseSmooth } from '../vendor/backend/signal/nodes/pose_smooth.js';
import { HandLandmarksToBones } from '../vendor/backend/signal/nodes/hand_landmarks_to_bones.js';
import { PoseMerge } from '../vendor/backend/signal/nodes/pose_merge.js';
import { BodyCalibration } from '../vendor/backend/signal/nodes/body_calibration.js';
import { HandHeightCompare } from '../vendor/backend/signal/nodes/hand_height_compare.js';
import {
  ARM_SOLVE_SWITCHES,
  HEAD_CALIB_BONES,
  HEAD_CALIB_PRESERVE,
  FINGER_CALIB_BONES,
  FINGER_MIRROR_PAIRS,
} from '../vendor/backend/mediapipe_tracker_graph.js';
import { BoneFilterBank } from '../vendor/frontend/oneEuroFilter.js';
import {
  BoneDynamicsBank,
  DEFAULT_POSE_DYNAMICS,
  type PoseDynamicsConfig,
} from '../vendor/frontend/secondOrderDynamics.js';

export type Landmark = { x: number; y: number; z: number; visibility?: number };

/** One MediaPipe Holistic frame, as vspark's CameraCapture sends it (already mirrored). */
export interface InputFrame {
  /** Capture time in milliseconds. */
  t: number;
  /** 33 pose world landmarks (visibility from the image-space set). */
  pose?: Landmark[] | null;
  /** 478 face landmarks (image space). */
  face?: Landmark[] | null;
  /** 21 hand world landmarks each. */
  leftHand?: Landmark[] | null;
  rightHand?: Landmark[] | null;
}

export type BoneRecord = Record<string, [number, number, number, number]>;
export interface OutputFrame {
  t: number;
  bones: BoneRecord;
}

/** Captured calibration offsets, as stored in vspark's body_calibration node state. */
export interface CalibrationOffsets {
  head?: BoneRecord;
  fingers?: BoneRecord;
}

export interface SolverOptions {
  /** Reproduce vspark's stale-stream behaviour (see header). Default true. */
  holdMissing?: boolean;
  /** Head behavior-config values. Defaults are vspark's graph defaults. */
  head?: { pitchGain?: number; yawGain?: number; rollGain?: number; restPitch?: number };
  /** Overrides for ARM_SOLVE_SWITCHES (`armSolve.*`); defaults are vspark's graph defaults. */
  armSolve?: Record<string, unknown>;
  /** Preloaded calibration offsets (e.g. exported from a vspark project). */
  calibration?: CalibrationOffsets;
}

function safePull(fn: Thunk<unknown>): unknown {
  try {
    return fn();
  } catch (err) {
    console.error('[baseline pull]', err);
    return undefined;
  }
}

/** Minimal stand-in for vspark's engine bind context: config fallbacks, wiring, in-memory state. */
function bind(
  node: Node,
  config: Record<string, unknown>,
  wires: Record<string, Thunk<unknown>> = {},
  lists: Record<string, Thunk<unknown>[]> = {}
): { outputs: Record<string, Thunk<unknown>>; handlers: Record<string, (p: unknown) => void> } {
  const outputs: Record<string, Thunk<unknown>> = {};
  const handlers: Record<string, (p: unknown) => void> = {};
  let state: unknown = undefined;
  const noopEmitter: Emitter<unknown> = { emit() {} };
  const ctx: NodeBindContext = {
    config,
    getState: <T>() => state as T,
    setState: (s) => {
      state = s;
    },
    makeEmitter: () => noopEmitter,
    // Engine semantics (vspark signal/engine.ts): a connected input pulls upstream and falls back
    // to config[port] when unconnected or when upstream yields undefined; a list input drops
    // undefined sources; a pull that throws yields undefined.
    valueThunk: (port) => () => {
      const v = wires[port] ? safePull(wires[port]) : undefined;
      return v === undefined ? config[port] : v;
    },
    listThunk: (port) => () =>
      (lists[port] ?? []).map(safePull).filter((v) => v !== undefined),
    registerOutputThunk: (port, fn) => {
      outputs[port] = fn;
    },
    registerHandler: (port, fn) => {
      handlers[port] = fn;
    },
    makeDynamicEmitter: () => noopEmitter,
    dynamicValueThunk: (port) => () => config[port],
    registerDynamicOutputs: () => {},
    isEnabled: () => config.enabled !== false,
  };
  node.bind(ctx);
  return { outputs, handlers };
}

/** The backend solver chain. Feed frames in time order; returns a pose for frames with pose data. */
export class BaselineSolver {
  private readonly holdMissing: boolean;
  private readonly inputs: {
    pose?: Landmark[];
    face?: Landmark[];
    leftHand?: Landmark[];
    rightHand?: Landmark[];
  } = {};
  private readonly out: Thunk<unknown>;
  private readonly calib: {
    head: Record<string, (p: unknown) => void>;
    fingers: Record<string, (p: unknown) => void>;
  };
  private readonly setCalibState: (c: CalibrationOffsets) => void;

  constructor(opts: SolverOptions = {}) {
    this.holdMissing = opts.holdMissing ?? true;
    const inp = this.inputs;
    const pose = () => inp.pose;
    const face = () => inp.face;
    const lh = () => inp.leftHand;
    const rh = () => inp.rightHand;

    const head = { pitchGain: 2.0, yawGain: 1.0, rollGain: 1.0, restPitch: -0.43, ...opts.head };
    const torso = bind(new PoseTorsoHeadToBones(), { ...head }, { pose, face });

    const armsCfg: Record<string, unknown> = { enabled: true }; // enabled = !useIk, useIk=false
    const smoothCfg: Record<string, unknown> = {};
    for (const [field, target, def] of ARM_SOLVE_SWITCHES) {
      const port = field === 'smoothing' ? 'enabled' : field;
      const value = opts.armSolve?.[field] ?? def;
      (target === 'pose_arms' ? armsCfg : smoothCfg)[port] = value;
    }
    const arms = bind(new PoseArmsToBones(), armsCfg, { pose, leftHand: lh, rightHand: rh });
    const smooth = bind(new PoseSmooth(), smoothCfg, {
      pose: arms.outputs.pose,
      confidence: arms.outputs.confidence,
    });
    const left = bind(new HandLandmarksToBones(), { side: 'left' }, { landmarks: lh });
    const right = bind(new HandLandmarksToBones(), { side: 'right' }, { landmarks: rh });
    // Same list order as the graph's edges into pose_merge.poses.
    const merge = bind(new PoseMerge(), {}, {}, {
      poses: [torso.outputs.pose, smooth.outputs.pose, left.outputs.pose, right.outputs.pose],
    });
    const handHeight = bind(new HandHeightCompare(), {}, { pose });

    const headCalibNode = new BodyCalibration();
    const headCalib = bind(
      headCalibNode,
      { boneFilter: HEAD_CALIB_BONES, preserveChildren: HEAD_CALIB_PRESERVE },
      { pose: merge.outputs.pose }
    );
    const fingerCalibNode = new BodyCalibration();
    const fingerCalib = bind(
      fingerCalibNode,
      { boneFilter: FINGER_CALIB_BONES, mirrorPairs: FINGER_MIRROR_PAIRS },
      { pose: headCalib.outputs.pose, mirrorSource: handHeight.outputs.side }
    );
    this.out = fingerCalib.outputs.pose;
    this.calib = { head: headCalib.handlers, fingers: fingerCalib.handlers };

    // Calibration state is node-private; reach it through the protected setState the same way
    // the engine restores persisted state.
    const setState = (n: Node, s: unknown) =>
      (n as unknown as { setState(s: unknown): void }).setState(s);
    this.setCalibState = (c) => {
      if (c.head) setState(headCalibNode, { bodyOffsets: c.head });
      if (c.fingers) setState(fingerCalibNode, { bodyOffsets: c.fingers });
    };
    if (opts.calibration) this.setCalibState(opts.calibration);
  }

  private withClock<T>(t: number, fn: () => T): T {
    const realNow = Date.now;
    Date.now = () => t;
    try {
      return fn();
    } finally {
      Date.now = realNow;
    }
  }

  private load(f: InputFrame): void {
    for (const k of ['face', 'leftHand', 'rightHand', 'pose'] as const) {
      const v = f[k];
      if (v && v.length) this.inputs[k] = v;
      else if (!this.holdMissing) this.inputs[k] = undefined;
    }
  }

  /** Process one frame. Returns undefined when the frame carries no pose (vspark emits nothing). */
  step(f: InputFrame): OutputFrame | undefined {
    this.load(f);
    if (!(f.pose && f.pose.length)) return undefined;
    const pose = this.withClock(f.t, () => this.out() as NormalizedPose | undefined);
    return pose ? { t: f.t, bones: pose.toRecord() } : undefined;
  }

  /** Fire the head ("Capture head neutral") or finger calibration capture on the current inputs. */
  capture(which: 'head' | 'fingers', t: number): void {
    this.withClock(t, () => this.calib[which].capture?.(undefined));
  }
  resetCalibration(which: 'head' | 'fingers'): void {
    this.calib[which].reset?.(undefined);
  }
}

export interface RenderOptions {
  /** Simulated render rate in Hz. Default 60. */
  hz?: number;
  /** One Euro parameters; defaults are vspark's BoneFilterBank defaults. */
  minCutoff?: number;
  beta?: number;
  /** Second-order dynamics (vspark per-avatar `poseDynamics`); disabled by default like vspark. */
  dynamics?: Partial<PoseDynamicsConfig>;
}

/**
 * Frontend stage: render ticks at a fixed rate from the first pose; each tick filters the latest
 * received pose with a per-bone One Euro filter (dt = render delta), like vspark's Viewport, which
 * re-filters the held pose every render frame. Tracking-loss fades, the animation blend and the
 * avatar compose levers are not reproduced.
 */
export function renderStage(poses: OutputFrame[], opts: RenderOptions = {}): OutputFrame[] {
  if (poses.length === 0) return [];
  const hz = opts.hz ?? 60;
  const dtMs = 1000 / hz;
  const dyn = { ...DEFAULT_POSE_DYNAMICS, ...opts.dynamics };
  const filters = new BoneFilterBank(opts.minCutoff ?? 1.0, opts.beta ?? 0.3);
  const dynamics = new BoneDynamicsBank();
  const q = new THREE.Quaternion();
  const out: OutputFrame[] = [];
  let i = 0;
  let lastNewIndex = -1;
  const end = poses[poses.length - 1].t;
  for (let t = poses[0].t; t <= end + 1e-9; t += dtMs) {
    while (i + 1 < poses.length && poses[i + 1].t <= t) i++;
    const newSample = i !== lastNewIndex;
    lastNewIndex = i;
    const bones: BoneRecord = {};
    for (const [name, r] of Object.entries(poses[i].bones)) {
      q.set(r[0], r[1], r[2], r[3]);
      let s = filters.filter(name, q, dtMs / 1000);
      if (dyn.enabled)
        s = dynamics.filter(name, s, dtMs / 1000, dyn.frequency, dyn.damping, dyn.response, newSample);
      bones[name] = [s.x, s.y, s.z, s.w];
    }
    out.push({ t, bones });
  }
  return out;
}
