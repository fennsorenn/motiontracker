/**
 * Class-instance signal-node model (Phase 2).
 *
 * A node is a live object whose decorated members ARE its ports:
 *
 *   class MyNode extends Node {
 *     @valueIn('a', 'Float')  a!: () => number;          // pull input  → thunk field
 *     @listIn('xs', 'Float')  xs!: () => number[];        // fan-in pull → thunk field
 *     @eventOut('out','String') out!: Emitter<string>;   // push output → emitter field
 *     @valueOut('p','Float')  p = () => this.a() * 2;     // pull output → thunk field
 *     @eventIn('fire','Trigger') onFire(p: Trigger) {     // push input  → handler method
 *       this.out.emit('hi');
 *     }
 *   }
 *
 * Transport is encoded by the decorator (and folded into the resolved type elsewhere);
 * there is no separate `kind` axis. State is reached through `this.getState/setState`
 * (engine-injected, DB-backed); nodes are otherwise stateless and rebuilt on reconcile.
 *
 * Port metadata is registered at CLASS-DEFINITION time (so the palette / NodeKindMeta
 * can be read without instantiating). Per-instance binding (assigning emitters/thunks,
 * subscribing handlers) happens when the engine calls `bind()`.
 */

import type { SignalTypeName } from './signal.js';
import type { ResolvedType, ResolvedPort } from './signal_types.js';

// ──────────────────────────────────────────────────────────────────────────────
// Port metadata — registered at definition time, keyed per node class
// ──────────────────────────────────────────────────────────────────────────────

export type PortDirection = 'in' | 'out';
export type PortTransport = 'event' | 'value' | 'list';

/** A statically-declared port (from a decorator). `member` is the field/method name. */
export interface PortMeta {
  name: string;
  direction: PortDirection;
  transport: PortTransport;
  /** Declared data-type tag (the leaf type before transport wrapping). */
  typeTag: SignalTypeName;
  /** The class member the decorator was attached to (field name or method name). */
  member: string;
}

/**
 * Port registration uses the Stage-3 decorator `ctx.metadata` object as a shared
 * per-class buffer: each member decorator pushes a `PortMeta` into
 * `ctx.metadata[PORT_BUFFER]`, and the `@SignalNode` class decorator (which runs
 * AFTER all member decorators and shares the same `ctx.metadata`) snapshots that
 * buffer into `PORT_META` keyed by the class constructor. This avoids depending on
 * `Symbol.metadata` being attached to the class at runtime (esbuild/tsx does not),
 * while still making ports readable WITHOUT instantiating the node.
 */
export const PORT_BUFFER = Symbol('vspark.portBuffer');

const PORT_META = new WeakMap<object, PortMeta[]>();

/** Push a port decl into the class's metadata buffer (called from member decorators). */
export function bufferPort(
  metadata: DecoratorMetadataObject | undefined,
  meta: PortMeta
): void {
  // `ctx.metadata` is always provided under Stage-3 decorators; guard for the type.
  if (!metadata) return;
  const md = metadata as Record<symbol, PortMeta[]>;
  (md[PORT_BUFFER] ??= []).push(meta);
}

/** Snapshot the metadata buffer onto the class (called from @SignalNode). */
export function harvestPorts(
  cls: object,
  metadata: DecoratorMetadataObject | undefined
): void {
  const buf =
    (metadata as Record<symbol, PortMeta[]> | undefined)?.[PORT_BUFFER] ?? [];
  // De-dupe on (direction, name); last decl wins.
  const byKey = new Map<string, PortMeta>();
  for (const p of buf) byKey.set(`${p.direction}\x00${p.name}`, p);
  PORT_META.set(cls, [...byKey.values()]);
}

/** Read a node class's declared ports. Returns [] if the class wasn't decorated. */
export function getPortMeta(cls: object): PortMeta[] {
  return PORT_META.get(cls) ?? [];
}

// ──────────────────────────────────────────────────────────────────────────────
// Emitter<T> — engine-provided instrumented push channel
//
// Nodes call `.emit(v)` on an @eventOut field. The engine creates the concrete
// instance (so emit can be instrumented for monitoring + enabled/try-catch), the
// node never `new`s one. This interface is what node code sees.
// ──────────────────────────────────────────────────────────────────────────────

export interface Emitter<T> {
  emit(value: T): void;
}

/** A pull source: the function the engine points a @valueIn/@listIn thunk field at. */
export type Thunk<T> = () => T;

// ──────────────────────────────────────────────────────────────────────────────
// Bind context — what the engine hands a node instance at wiring time
// ──────────────────────────────────────────────────────────────────────────────

export interface NodeBindContext {
  /** Live config object for this node (defaultConfig merged with stored overrides). */
  readonly config: Record<string, unknown>;
  /** Read this node's persisted state. */
  getState<T = unknown>(): T;
  /** Persist new state for this node. */
  setState(state: unknown): void;
  /** Provision an instrumented emitter for an @eventOut port (engine-instrumented). */
  makeEmitter(portName: string): Emitter<unknown>;
  /** Resolve the pull-thunk for a @valueIn port (upstream output, or config fallback). */
  valueThunk(portName: string): Thunk<unknown>;
  /** Resolve the gather-thunk for a @listIn port (all connected sources → array). */
  listThunk(portName: string): Thunk<unknown[]>;
  /** Register a node's @valueOut thunk so downstream pulls reach it. */
  registerOutputThunk(portName: string, fn: Thunk<unknown>): void;
  /** Subscribe an @eventIn handler so upstream emits reach it. */
  registerHandler(portName: string, fn: (payload: unknown) => void): void;
  /** Provision an instrumented emitter for a DYNAMIC event-output (emitOn). */
  makeDynamicEmitter(portName: string): Emitter<unknown>;
  /** Resolve a DYNAMIC value-input pull-thunk by name (input()). */
  dynamicValueThunk(portName: string): Thunk<unknown>;
  /** Register a resolver for DYNAMIC value (pull) outputs (setDynamicOutputs). */
  registerDynamicOutputs(resolve: (portName: string) => unknown): void;
  /** Whether this node is currently enabled (config.enabled !== false). */
  isEnabled(): boolean;
}

// ──────────────────────────────────────────────────────────────────────────────
// Node — abstract base
// ──────────────────────────────────────────────────────────────────────────────

export abstract class Node {
  /** Set by `bind()`; node code reaches config/state through the helpers below. */
  private _ctx?: NodeBindContext;

  /** Per-instance dynamic emitters created via emitOn (kept so we reuse one per name). */
  private _dynEmitters = new Map<string, Emitter<unknown>>();

  // ── lifecycle ──────────────────────────────────────────────────────────────

  /**
   * Engine entry point, called once after construction. Reads the class's declared
   * ports (harvested at definition time) and binds each by transport:
   *   - eventOut → assign an instrumented Emitter to the field
   *   - valueIn  → assign the upstream pull-thunk (or config fallback) to the field
   *   - listIn   → assign the gather-thunk to the field
   *   - valueOut → register the field's thunk (the node's initial value) for downstream pulls
   *   - eventIn  → subscribe the handler method to upstream emits
   * Capture each @valueOut field's initial thunk BEFORE overwriting any fields, so a
   * node whose output thunk closes over an input field still sees the bound thunk
   * (the thunk reads `this.<field>` lazily at pull time, after bind completes).
   */
  bind(ctx: NodeBindContext): void {
    this._ctx = ctx;
    const ports = getPortMeta(this.constructor);
    const self = this as unknown as Record<string, unknown>;

    for (const p of ports) {
      if (p.direction === 'out' && p.transport === 'event') {
        self[p.member] = ctx.makeEmitter(p.name);
      } else if (p.direction === 'in' && p.transport === 'value') {
        self[p.member] = ctx.valueThunk(p.name);
      } else if (p.direction === 'in' && p.transport === 'list') {
        self[p.member] = ctx.listThunk(p.name);
      }
    }
    for (const p of ports) {
      if (p.direction === 'out' && p.transport === 'value') {
        const fn = self[p.member] as Thunk<unknown>;
        ctx.registerOutputThunk(p.name, () => fn.call(this));
      } else if (p.direction === 'in' && p.transport === 'event') {
        const method = self[p.member] as (payload: unknown) => void;
        ctx.registerHandler(p.name, (payload) => method.call(this, payload));
      }
    }

    // Post-bind hook (ctx is now available — safe for dynamic-output registration etc.).
    this.onBind();
  }

  /**
   * Override for setup that needs `this.config`/state/dynamic accessors — runs once
   * after `bind()` has wired all ports. (The constructor runs BEFORE bind, so dynamic
   * registration must happen here, not in the constructor.) Default no-op.
   */
  protected onBind(): void {}

  /**
   * Called by the engine when the graph is torn down (stop / reconcile). Override
   * to release external resources a node acquired — e.g. `set_data` clears the
   * data-channel entries it published so they don't linger after the graph stops.
   * Default no-op.
   */
  unbind(): void {
    this.onUnbind();
  }

  /** Override target for teardown cleanup. Runs once on graph dispose. */
  protected onUnbind(): void {}

  // ── state (engine-injected, DB-backed) ───────────────────────────────────────

  protected getState<T = unknown>(): T {
    return this._ctx!.getState<T>();
  }
  protected setState(state: unknown): void {
    this._ctx!.setState(state);
  }
  protected get config(): Record<string, unknown> {
    return this._ctx!.config;
  }
  protected get enabled(): boolean {
    return this._ctx!.isEnabled();
  }

  // ── dynamic ports (used only by generic nodes: pack_event / unpack_event) ─────

  /** Pull a dynamic value-input by port name (the dynamic counterpart of @valueIn). */
  protected input(portName: string): unknown {
    return this._ctx!.dynamicValueThunk(portName)();
  }
  /** Push a value on a dynamic event-output by port name (counterpart of @eventOut). */
  protected emitOn(portName: string, value: unknown): void {
    let em = this._dynEmitters.get(portName);
    if (!em) {
      em = this._ctx!.makeDynamicEmitter(portName);
      this._dynEmitters.set(portName, em);
    }
    em.emit(value);
  }

  /**
   * Register a resolver for DYNAMIC value (pull) outputs — ports that don't exist as
   * decorated members and whose set/types are computed by inferPorts. When a downstream
   * pulls such a port, the engine calls this resolver with the port name. Used by the
   * generic `unpack_event` whose per-field outputs are pulled (not pushed), so existing
   * pull-based pipelines (VMC/lipsync/mediapipe) keep working unchanged.
   */
  protected setDynamicOutputs(resolve: (portName: string) => unknown): void {
    this._ctx!.registerDynamicOutputs(resolve);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// inferPorts contract — a node class may compute its actual ports from connections
// ──────────────────────────────────────────────────────────────────────────────

export interface InferCtx {
  /** Resolved type currently wired into each input port (by port name). */
  resolvedInputs: Record<string, ResolvedType>;
  /** This node's live config. */
  config: unknown;
  /** Owner kind of the graph this node lives in (`project` / `scene_node` /
   *  `compose_layer`), when known. Scope-aware nodes (e.g. `scene_entity`) use it
   *  to vary their port types. Undefined for graphs with no scope context. */
  ownerKind?: import('./types.js').LogicOwnerKind;
}

export interface InferResult {
  inputPorts: ResolvedPort[];
  outputPorts: ResolvedPort[];
}
