/**
 * Signal graph type system.
 *
 * Data types, the Event<T> push-signal wrapper, typed port machinery,
 * the @SignalNode class decorator, and graph descriptor types.
 */
import { byId, itemsOf, type IdMap } from './idMap.js';

// ──────────────────────────────────────────────────────────────────────────────
// VRM bone name registry (VRM 1.x humanoid spec)
// ──────────────────────────────────────────────────────────────────────────────

export const VRM_BONE_NAMES = [
  'hips',
  'spine',
  'chest',
  'upperChest',
  'neck',
  'head',
  'jaw',
  'leftEye',
  'rightEye',
  'leftShoulder',
  'leftUpperArm',
  'leftLowerArm',
  'leftHand',
  'rightShoulder',
  'rightUpperArm',
  'rightLowerArm',
  'rightHand',
  'leftUpperLeg',
  'leftLowerLeg',
  'leftFoot',
  'leftToes',
  'rightUpperLeg',
  'rightLowerLeg',
  'rightFoot',
  'rightToes',
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

export type VRMBoneName = (typeof VRM_BONE_NAMES)[number];

// ──────────────────────────────────────────────────────────────────────────────
// Quaternion — immutable unit quaternion with full algebra
// ──────────────────────────────────────────────────────────────────────────────

export class Quaternion {
  constructor(
    public readonly x: number,
    public readonly y: number,
    public readonly z: number,
    public readonly w: number
  ) {}

  static readonly IDENTITY = new Quaternion(0, 0, 0, 1);

  static fromArray(a: readonly [number, number, number, number]): Quaternion {
    return new Quaternion(a[0], a[1], a[2], a[3]);
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  get magnitudeSquared(): number {
    return (
      this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w
    );
  }

  get isValid(): boolean {
    return this.magnitudeSquared > 1e-9;
  }

  normalize(): Quaternion {
    const len = Math.sqrt(this.magnitudeSquared);
    if (len < 1e-9) return Quaternion.IDENTITY;
    return new Quaternion(
      this.x / len,
      this.y / len,
      this.z / len,
      this.w / len
    );
  }

  invert(): Quaternion {
    const m2 = this.magnitudeSquared;
    if (m2 < 1e-9) return Quaternion.IDENTITY;
    const s = 1 / m2;
    return new Quaternion(-this.x * s, -this.y * s, -this.z * s, this.w * s);
  }

  multiply(rhs: Quaternion): Quaternion {
    const { x: ax, y: ay, z: az, w: aw } = this;
    const { x: bx, y: by, z: bz, w: bw } = rhs;
    return new Quaternion(
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz
    );
  }

  premultiply(lhs: Quaternion): Quaternion {
    return lhs.multiply(this);
  }

  /**
   * Spherical linear interpolation towards `to` by `t` (clamped to [0, 1]),
   * along the shortest arc. Falls back to a normalized lerp when the two
   * rotations are nearly parallel (where the sine denominator is unstable).
   */
  slerp(to: Quaternion, t: number): Quaternion {
    const k = t <= 0 ? 0 : t >= 1 ? 1 : t;
    if (k === 0) return this;
    if (k === 1) return to;

    const a = this.normalize();
    let b = to.normalize();
    let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    // Shortest arc: flip the destination when the rotations point apart.
    if (dot < 0) {
      b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
      dot = -dot;
    }

    if (dot > 0.9995) {
      // Nearly parallel — lerp + renormalize (slerp's denominator → 0 here).
      return new Quaternion(
        a.x + (b.x - a.x) * k,
        a.y + (b.y - a.y) * k,
        a.z + (b.z - a.z) * k,
        a.w + (b.w - a.w) * k
      ).normalize();
    }

    const theta = Math.acos(dot);
    const sinTheta = Math.sin(theta);
    const wa = Math.sin((1 - k) * theta) / sinTheta;
    const wb = Math.sin(k * theta) / sinTheta;
    return new Quaternion(
      a.x * wa + b.x * wb,
      a.y * wa + b.y * wb,
      a.z * wa + b.z * wb,
      a.w * wa + b.w * wb
    );
  }

  /**
   * Build a unit quaternion from intrinsic ZYX Euler angles (radians):
   * Rz(roll) · Ry(yaw) · Rx(pitch). pitch = X axis, yaw = Y axis, roll = Z axis.
   * Matches the `euler_to_quaternion` signal node convention; `toEuler` is its inverse.
   */
  static fromEuler(pitch: number, yaw: number, roll: number): Quaternion {
    const cx = Math.cos(pitch / 2),
      sx = Math.sin(pitch / 2);
    const cy = Math.cos(yaw / 2),
      sy = Math.sin(yaw / 2);
    const cz = Math.cos(roll / 2),
      sz = Math.sin(roll / 2);
    return new Quaternion(
      sx * cy * cz - cx * sy * sz,
      cx * sy * cz + sx * cy * sz,
      cx * cy * sz - sx * sy * cz,
      cx * cy * cz + sx * sy * sz
    ).normalize();
  }

  /**
   * Decompose into intrinsic ZYX Euler angles (radians), the inverse of `fromEuler`.
   * Returns `{ pitch, yaw, roll }` (X, Y, Z). Near the yaw = ±90° singularity
   * (gimbal lock) the roll/pitch split is ambiguous; this collapses the coupled
   * rotation onto `roll` and returns `pitch = 0` — acceptable for manual calibration.
   */
  toEuler(): { pitch: number; yaw: number; roll: number } {
    const { x, y, z, w } = this.normalize();
    const sinYaw = 2 * (w * y - x * z);
    if (Math.abs(sinYaw) > 0.99999) {
      // Gimbal lock: yaw at ±90°, pitch and roll are coupled.
      return {
        pitch: 0,
        yaw: Math.sign(sinYaw) * (Math.PI / 2),
        roll: Math.atan2(2 * (x * y - w * z), 1 - 2 * (x * x + z * z)),
      };
    }
    return {
      pitch: Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
      yaw: Math.asin(sinYaw),
      roll: Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    };
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// BoneRotations — raw VMC input, keyed by source app bone name
// ──────────────────────────────────────────────────────────────────────────────

export class BoneRotations {
  private readonly _bones: Map<string, Quaternion>;

  constructor(entries: Iterable<readonly [string, Quaternion]> = []) {
    this._bones = new Map(entries);
  }

  static fromRecord(
    rec: Record<string, readonly [number, number, number, number]>
  ): BoneRotations {
    return new BoneRotations(
      Object.entries(rec).map(([k, v]) => [k, Quaternion.fromArray(v)] as const)
    );
  }

  get(bone: string): Quaternion | undefined {
    return this._bones.get(bone);
  }
  has(bone: string): boolean {
    return this._bones.has(bone);
  }
  get size(): number {
    return this._bones.size;
  }
  entries(): IterableIterator<[string, Quaternion]> {
    return this._bones.entries();
  }
  keys(): IterableIterator<string> {
    return this._bones.keys();
  }

  map(fn: (q: Quaternion, bone: string) => Quaternion): BoneRotations {
    return new BoneRotations(
      Array.from(this._bones.entries()).map(([b, q]) => [b, fn(q, b)] as const)
    );
  }

  filter(fn: (bone: string, q: Quaternion) => boolean): BoneRotations {
    return new BoneRotations(
      Array.from(this._bones.entries()).filter(([b, q]) => fn(b, q))
    );
  }

  toRecord(): Record<string, [number, number, number, number]> {
    const out: Record<string, [number, number, number, number]> = {};
    for (const [b, q] of this._bones) out[b] = q.toArray();
    return out;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// NormalizedPose — VRM-mapped, coordinate-corrected bone rotations
// ──────────────────────────────────────────────────────────────────────────────

/** A bone translation offset, `[x, y, z]`, as a fraction of the avatar's hip height. */
export type BoneOffset = readonly [number, number, number];

export class NormalizedPose {
  private readonly _bones: Map<VRMBoneName, Quaternion>;
  /**
   * Optional per-bone TRANSLATION, carried alongside the rotations.
   *
   * The pose pipeline is rotation-first and every mapper, calibration and merge
   * step only ever touches `_bones`; offsets ride along untouched through `map`
   * and `with` so none of them had to learn about translation. Currently only the
   * hips are ever offset (by the stylizer's body-shift channels), and the unit is
   * deliberately RELATIVE — a fraction of the avatar's own hip height — so a rig
   * authored on one model reads the same on a taller or shorter one.
   */
  private readonly _offsets: Map<VRMBoneName, BoneOffset>;

  constructor(
    entries: Iterable<readonly [VRMBoneName, Quaternion]> = [],
    offsets: Iterable<readonly [VRMBoneName, BoneOffset]> = []
  ) {
    this._bones = new Map(entries);
    this._offsets = new Map(offsets);
  }

  /** This bone's translation offset, if the pose carries one. */
  offset(bone: VRMBoneName): BoneOffset | undefined {
    return this._offsets.get(bone);
  }
  offsetEntries(): IterableIterator<[VRMBoneName, BoneOffset]> {
    return this._offsets.entries();
  }
  get offsetCount(): number {
    return this._offsets.size;
  }

  /** Copy with one bone's translation offset set (rotations untouched). */
  withOffset(bone: VRMBoneName, value: BoneOffset): NormalizedPose {
    const next = new Map(this._offsets);
    next.set(bone, value);
    return new NormalizedPose(this._bones, next);
  }

  /** Wire form for the offsets. Empty when the pose is rotation-only. */
  offsetsToRecord(): Record<string, [number, number, number]> {
    const out: Record<string, [number, number, number]> = {};
    for (const [b, v] of this._offsets) out[b] = [v[0], v[1], v[2]];
    return out;
  }

  get(bone: VRMBoneName): Quaternion | undefined {
    return this._bones.get(bone);
  }
  has(bone: VRMBoneName): boolean {
    return this._bones.has(bone);
  }
  get size(): number {
    return this._bones.size;
  }
  entries(): IterableIterator<[VRMBoneName, Quaternion]> {
    return this._bones.entries();
  }
  keys(): IterableIterator<VRMBoneName> {
    return this._bones.keys();
  }

  with(bone: VRMBoneName, q: Quaternion): NormalizedPose {
    const next = new Map(this._bones);
    next.set(bone, q);
    return new NormalizedPose(next, this._offsets);
  }

  map(fn: (q: Quaternion, bone: VRMBoneName) => Quaternion): NormalizedPose {
    return new NormalizedPose(
      Array.from(this._bones.entries()).map(([b, q]) => [b, fn(q, b)] as const),
      this._offsets
    );
  }

  toRecord(): Record<string, [number, number, number, number]> {
    const out: Record<string, [number, number, number, number]> = {};
    for (const [b, q] of this._bones) out[b] = q.toArray();
    return out;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Blendshapes — VRM expression weights, clamped to [0, 1]
// ──────────────────────────────────────────────────────────────────────────────

export class Blendshapes {
  private readonly _values: Map<string, number>;

  constructor(entries: Iterable<readonly [string, number]> = []) {
    this._values = new Map(entries);
  }

  static fromRecord(rec: Record<string, number>): Blendshapes {
    return new Blendshapes(Object.entries(rec));
  }

  get(name: string): number {
    return this._values.get(name) ?? 0;
  }
  has(name: string): boolean {
    return this._values.has(name);
  }
  get size(): number {
    return this._values.size;
  }
  entries(): IterableIterator<[string, number]> {
    return this._values.entries();
  }

  with(name: string, value: number): Blendshapes {
    const next = new Map(this._values);
    next.set(name, Math.max(0, Math.min(1, value)));
    return new Blendshapes(next);
  }

  map(fn: (value: number, name: string) => number): Blendshapes {
    return new Blendshapes(
      Array.from(this._values.entries()).map(([n, v]) => [n, fn(v, n)] as const)
    );
  }

  toRecord(): Record<string, number> {
    return Object.fromEntries(this._values);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// InterceptorFrame — opaque token passed through the pose interceptor chain
// ──────────────────────────────────────────────────────────────────────────────

export interface InterceptorFrame {
  /** Scene node the broadcast is addressed to. */
  readonly nodeId: string;
  /** Pose at the point this interceptor was invoked. */
  readonly pose: NormalizedPose;
  /** Priority of the on_pose_broadcast node that produced this frame. */
  readonly priority: number;
}

/**
 * The blendshape counterpart of `InterceptorFrame` — passed through the
 * blendshape interceptor chain (see the Expression Limits behavior). Kept as a
 * separate type (rather than widening `InterceptorFrame`) so a pose frame can
 * never be wired into a blendshape terminal.
 */
export interface BlendshapeInterceptorFrame {
  /** Scene node the broadcast is addressed to. */
  readonly nodeId: string;
  /** Blendshapes at the point this interceptor was invoked. */
  readonly blendshapes: Blendshapes;
  /** Priority of the on_blendshapes_broadcast node that produced this frame. */
  readonly priority: number;
}

// ──────────────────────────────────────────────────────────────────────────────
// PoseFrame — fully assembled frame, wire format for server → client broadcast
// ──────────────────────────────────────────────────────────────────────────────

export class PoseFrame {
  constructor(
    public readonly behaviorId: string,
    public readonly pose: NormalizedPose,
    public readonly blendshapes: Blendshapes,
    public readonly timestamp: number
  ) {}

  toWire(): {
    bones: Record<string, [number, number, number, number]>;
    blendshapes: Record<string, number>;
    offsets?: Record<string, [number, number, number]>;
  } {
    return {
      bones: this.pose.toRecord(),
      blendshapes: this.blendshapes.toRecord(),
      // Omitted entirely for rotation-only poses, which is nearly all of them.
      ...(this.pose.offsetCount > 0
        ? { offsets: this.pose.offsetsToRecord() }
        : {}),
    };
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Event<T> — push signal wrapper
// ──────────────────────────────────────────────────────────────────────────────

export interface Event<T> {
  readonly payload: T;
  readonly timestamp: number;
}

export function mkEvent<T>(payload: T, timestamp = Date.now()): Event<T> {
  return { payload, timestamp };
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal type system — port types and connection safety
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Per-bone tracking confidence in [0, 1], keyed by VRM bone name. A bone absent from the map
 * counts as fully confident. Produced by converters alongside their pose (e.g. from landmark
 * visibility) and consumed by `pose_smooth`.
 */
export type BoneConfidence = Partial<Record<VRMBoneName, number>>;

export interface SignalTypeMap {
  BoneRotations: BoneRotations;
  NormalizedPose: NormalizedPose;
  Blendshapes: Blendshapes;
  /** Raw ARKit 52-shape weights before any VRM expression mapping. */
  ArkitBlendshapes: Blendshapes;
  PoseFrame: PoseFrame;
  Float: number;
  Bool: boolean;
  Trigger: void;
  /** General-purpose string value (IDs, names). */
  String: string;
  /** A component's full config JSON, pulled as a value. */
  BehaviorConfig: Record<string, unknown>;
  /**
   * Reference to a scene NODE — carried as the node's bare id string at runtime.
   * The narrow counterpart of `SceneEntity` (a `SceneNode` is assignable to a
   * `SceneEntity` input, but not vice versa). See `isAssignable`.
   */
  SceneNode: string;
  /** Reference to a compose LAYER — its bare id string at runtime. Narrow
   *  counterpart of `SceneEntity`, like `SceneNode`. */
  ComposeLayer: string;
  /** ARKit→target mapping table: shape name → [(targetName, weight), ...] */
  MappingTable: Record<string, [string, number][]> | null;
  /** Opaque token passed through the pose interceptor chain. */
  InterceptorFrame: InterceptorFrame;
  /** Opaque token passed through the blendshape interceptor chain. */
  BlendshapeInterceptorFrame: BlendshapeInterceptorFrame;
  /** A single unit quaternion rotation. */
  Quaternion: Quaternion;
  /**
   * One frame of stylized-tracking drivers — the low-dimensional performance
   * summary (head/body orientation, arm height, energy) that the `pose_stylizer`
   * behavior fans back out across the whole body. See `style_rig.ts`.
   */
  StyleDrivers: import('./style_rig.js').StyleDrivers;
  /** Wildcard — compatible with any other type for generic nodes. */
  Any: unknown;
  /** Raw MediaPipe landmark array (face=478, hand=21, pose=33 points). */
  LandmarkList: Array<{ x: number; y: number; z: number; visibility?: number }>;
  /** IK end-effector targets for a single frame, ready for frontend solve. */
  IkTargets: import('./types.js').IkTargetFrame;
  /** Per-bone tracking confidence — see `BoneConfidence`. */
  BoneConfidence: BoneConfidence;
  /**
   * Reference to a configured overlive login account (Twitch / StreamElements).
   * Carried as the account row id string at runtime; the frontend port editor
   * renders a dropdown of the project's accounts when nothing is wired in.
   */
  Account: string;
  /**
   * Payload emitted by `spawn_clip`. Carries the ids of the ephemeral entity
   * that was cloned and the ephemeral clip duplicate now playing on it, plus
   * which kind of entity was spawned. `set_*_param` nodes accept a
   * `Event<SpawnRef>` on a `spawnRef` input to retarget the call to the
   * tmp entity for that fire. See dev-notes/modules/spawn.md.
   */
  SpawnRef: {
    tmpNodeId: string;
    tmpClipId: string;
    kind: 'scene_node' | 'compose_layer';
  };
  /**
   * Reference to a scene entity — EITHER a scene node or a compose layer —
   * carried as that entity's bare id string at runtime (ids are unique across
   * both, so the kind isn't needed at runtime). The generic supertype: a
   * `SceneNode` or `ComposeLayer` output is assignable into a `SceneEntity`
   * input (see `isAssignable`). Used as the `scope` input on `set_data` to target
   * which consumer a published field-set is visible to (the consumer listens on
   * its own id). The port editor renders a node/layer dropdown when nothing is
   * wired in.
   */
  SceneEntity: string;
}

export type SignalTypeName = keyof SignalTypeMap;

// ──────────────────────────────────────────────────────────────────────────────
// SignalNodeClass — structural interface expected by the engine (Phase 2)
//
// A node class is a constructor for a `Node` subclass (see node.ts), carrying a
// static `kind`. Ports are declared as decorated members and harvested at
// class-definition time by `@SignalNode` (no static inputPorts/outputPorts arrays).
// An optional static `inferPorts` lets a node compute its resolved ports from its
// connected inputs; ordinary nodes omit it and fall back to lifting their static
// port declarations.
// ──────────────────────────────────────────────────────────────────────────────

import type { Node, InferCtx, InferResult } from './node.js';
import { harvestPorts } from './node.js';
import type { ResolvedType, Transport } from './signal_types.js';

export interface SignalNodeClass {
  readonly kind: string;
  /**
   * Opt in to DURABLE state: `this.setState(...)` is written through to the graph
   * owner's store (a behavior's `config._nodeState` / a logic row's `node_state`)
   * and restored on the next boot. Only for state a user deliberately produced and
   * expects to survive a restart — calibration offsets, captured neutral poses.
   *
   * Omitted (the default) means SCRATCH state: it lives in the running graph and is
   * dropped on teardown. Every per-frame node belongs here. Scratch state must never
   * be persisted, for two reasons: it churns a DB write per frame, and it round-trips
   * through JSON, which strips class identity — a `Blendshapes`/`NormalizedPose`
   * comes back as a plain object and blows up the first consumer that calls a method
   * on it.
   */
  readonly persistState?: boolean;
  /** Construct a fresh node instance (the engine then calls `instance.bind(...)`). */
  new (): Node;
  /** Optional shape inference from connected inputs (pack/unpack/queue). */
  inferPorts?(ctx: InferCtx): InferResult;
}

// ──────────────────────────────────────────────────────────────────────────────
// @SignalNode decorator — attaches display metadata to a node class
//
// Metadata is stored in a module-level WeakMap keyed by the class constructor.
// getNodeDisplay(cls) retrieves it without any Symbol.metadata dependency.
// ──────────────────────────────────────────────────────────────────────────────

/** Display properties rendered in the React Flow graph editor. */
export interface NodeDisplay {
  /** Human-readable node label shown in the canvas. */
  label: string;
  /** Short description shown in the node palette tooltip. */
  description?: string;
  /** Category tags — a node appears in every tab whose tag it carries. */
  tags: string[];
  /** Hex fill colour for the node header in the canvas. */
  color: string;
  /** If true the node is auto-wired by the system and hidden from the user palette. */
  internal?: boolean;
}

/** Per-SignalTypeName edge/handle colour used in the React Flow canvas. */
export const SIGNAL_TYPE_COLORS: Record<SignalTypeName, string> = {
  BoneRotations: '#6a9fb5',
  NormalizedPose: '#5ba45b',
  Blendshapes: '#b58a3a',
  ArkitBlendshapes: '#c47a20',
  PoseFrame: '#7a5baf',
  Float: '#a0a0a0',
  Bool: '#c87070',
  Trigger: '#888888',
  String: '#7ab8c8',
  BehaviorConfig: '#8a6aaf',
  SceneNode: '#6a8aaf',
  ComposeLayer: '#6aaf9a',
  MappingTable: '#a07050',
  InterceptorFrame: '#9a5a8a',
  BlendshapeInterceptorFrame: '#b5708a',
  Quaternion: '#5a9a7a',
  StyleDrivers: '#b06a9a',
  LandmarkList: '#7a9a6a',
  IkTargets: '#a06a9a',
  BoneConfidence: '#8aa05a',
  Account: '#9146ff',
  SpawnRef: '#c97a3a',
  SceneEntity: '#4aa0a0',
  Any: '#888888',
};

const _displayMap = new WeakMap<object, NodeDisplay>();

/**
 * Class decorator. Co-locates display metadata with the node definition AND harvests
 * the port declarations buffered by the member decorators (`@eventIn`/`@valueIn`/…).
 * It must run after the member decorators (class decorators always do) and shares the
 * same `ctx.metadata` buffer, so by the time it runs every port is registered.
 *
 * ```ts
 * @SignalNode({ label: 'Body Calibration', tags: ['calibration'], color: '#4a6a9f' })
 * export class BodyCalibration extends Node { ... }
 * ```
 */
export function SignalNode(display: NodeDisplay) {
  return function (cls: object, ctx: ClassDecoratorContext): void {
    _displayMap.set(cls, display);
    harvestPorts(cls, ctx.metadata);
  };
}

/** Returns the NodeDisplay attached by @SignalNode, or undefined if not decorated. */
export function getNodeDisplay(cls: object): NodeDisplay | undefined {
  return _displayMap.get(cls);
}

// ──────────────────────────────────────────────────────────────────────────────
// Graph descriptor — shared data shape for both implicit (code-defined) and
// explicit (DB-persisted) graphs. The engine instantiates from it; the React
// Flow canvas renders it.
// ──────────────────────────────────────────────────────────────────────────────

export interface GraphNodeDescriptor {
  /** Unique within the graph. */
  id: string;
  /** Must match a registered SignalNodeClass.kind. */
  kind: string;
  /** Canvas position for React Flow. */
  position: { x: number; y: number };
  /** Default config values, merged with any user-stored nodeConfig overrides. */
  defaultConfig?: Record<string, unknown>;
}

export interface GraphEdgeDescriptor {
  fromNodeId: string;
  fromPort: string;
  toNodeId: string;
  toPort: string;
  /** Defaults to 'event'. Value = pulled once per (to, port). List = multiple sources fan into an array. */
  kind?: 'event' | 'value' | 'list';
}

export interface GraphDescriptor {
  /** Unique across all graphs in the system. */
  id: string;
  label: string;
  /** Whether the graph topology is editable by the user. */
  readonly: boolean;
  nodes: GraphNodeDescriptor[];
  edges: GraphEdgeDescriptor[];
}

// ──────────────────────────────────────────────────────────────────────────────
// NodeKindMeta — serialisable node kind descriptor served by the API
// Used by the frontend node palette and canvas to render without importing
// backend node classes directly.
// ──────────────────────────────────────────────────────────────────────────────

/** Live state of a single node, returned by the monitoring API. */
export interface NodeStateSnapshot {
  lastExecutedAt: number | null;
  /** Last pulled / emitted scalar values (value ports + small event payloads). */
  portValues: Record<string, unknown>;
  /** Current config object for the node. */
  config: unknown;
}

/** Live state of a single event edge. Key format: `fromId:fromPort:toId:toPort`. */
export interface EdgeStateSnapshot {
  lastFiredAt: number | null;
  lastValue: unknown;
}

/** Combined graph monitoring snapshot. */
export interface GraphStateSnapshot {
  nodes: Record<string, NodeStateSnapshot>;
  edges: Record<string, EdgeStateSnapshot>;
}

/**
 * A node port as served to the frontend. Carries the port's STATIC declared shape:
 * its resolved type (transport folded in) plus the leaf type tag + transport for
 * rendering. The editor recomputes per-instance dynamic ports via shared inference;
 * this is the per-kind baseline (what `defaultInfer` would produce).
 */
export interface NodePortMeta {
  name: string;
  /** Resolved structural type (transport folded into the constructor). */
  resolved: ResolvedType;
  /** Leaf data-type tag (for colour lookup in SIGNAL_TYPE_COLORS). */
  typeTag: SignalTypeName;
  /** Derived transport: 'event' | 'value' | 'list'. */
  transport: Transport;
}

export interface NodeKindMeta {
  kind: string;
  inputPorts: NodePortMeta[];
  outputPorts: NodePortMeta[];
  display: NodeDisplay | undefined;
  /** True if the kind has a custom inferPorts (ports may grow/shrink at edit time). */
  dynamic?: boolean;
}

// ──────────────────────────────────────────────────────────────────────────────
// Descriptor document form — how a persisted graph is SYNCED
// ──────────────────────────────────────────────────────────────────────────────

/**
 * A graph descriptor as a synced document: nodes and edges keyed by id rather
 * than listed.
 *
 * `GraphDescriptor` above stays the runtime shape — it is what the engine
 * instantiates from and what the canvas renders, and both want lists. But a
 * list is ONE mesh path: with `nodes` as an array, two people moving different
 * nodes of one graph wrote the same path and last-writer-wins threw one edit
 * away, taking the whole graph with it. Keyed, an edit addresses
 * `descriptor.nodes.<nodeId>.position` and the two merge.
 *
 * The two forms convert at the boundary ({@link toGraphDescriptor} /
 * {@link toDescriptorDoc}), so nothing downstream of the canvas or the engine
 * has to know this exists.
 */
export interface GraphDescriptorDoc {
  id: string;
  label: string;
  readonly: boolean;
  nodes: IdMap<GraphNodeDescriptor>;
  edges: IdMap<GraphEdgeDescriptor>;
}

/** An edge's key: DERIVED from its endpoints, never minted.
 *
 *  Edges carry no id of their own, and deriving one means two peers drawing the
 *  same connection converge on one key — an LWW race on identical content —
 *  instead of minting two uuids and leaving the graph with a duplicate edge
 *  nobody can see. Same format the node-state snapshot uses for event edges. */
export const edgeKey = (e: GraphEdgeDescriptor): string =>
  `${e.fromNodeId}:${e.fromPort}:${e.toNodeId}:${e.toPort}`;

/** Document → runtime. Accepts the list form too, so a descriptor stored before
 *  the keying (or posted by an outside service) still loads. Nodes come back in
 *  id order and edges in key order: the engine resolves by id and the canvas by
 *  position, so neither depends on the order, but a stable one keeps diffs and
 *  snapshots readable. */
export function toGraphDescriptor(
  d: GraphDescriptorDoc | GraphDescriptor | null | undefined
): GraphDescriptor {
  const src = (d ?? {}) as Partial<GraphDescriptorDoc & GraphDescriptor>;
  const nodes = Array.isArray(src.nodes)
    ? (src.nodes as GraphNodeDescriptor[])
    : itemsOf(src.nodes as IdMap<GraphNodeDescriptor> | undefined).sort((a, b) =>
        a.id.localeCompare(b.id)
      );
  const edges = Array.isArray(src.edges)
    ? (src.edges as GraphEdgeDescriptor[])
    : itemsOf(src.edges as IdMap<GraphEdgeDescriptor> | undefined).sort((a, b) =>
        edgeKey(a).localeCompare(edgeKey(b))
      );
  return {
    id: src.id ?? '',
    label: src.label ?? '',
    readonly: src.readonly ?? false,
    nodes,
    edges,
  };
}

/** Runtime → document. For writing a whole descriptor at once (a create, a
 *  preset import); an ordinary edit writes the one element it changes. */
export function toDescriptorDoc(d: GraphDescriptor): GraphDescriptorDoc {
  const edges: IdMap<GraphEdgeDescriptor> = {};
  for (const e of d.edges ?? []) edges[edgeKey(e)] = e;
  return {
    id: d.id,
    label: d.label,
    readonly: d.readonly ?? false,
    nodes: byId(d.nodes ?? []),
    edges,
  };
}
