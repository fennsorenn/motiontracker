/**
 * Structural signal-graph type system (Phase 2).
 *
 * Replaces the old two-axis port model (a transport `kind` + a flat `SignalTypeName`
 * tag) with a single structural `ResolvedType` AST in which **transport is folded into
 * the type constructor**:
 *
 *   - `event`  ⇒ push    (Event<T> — the engine subscribes a handler)
 *   - `list`   ⇒ pull fan-in (List<T> — many sources gathered into an array)
 *   - anything else ⇒ pull (a single synchronous value)
 *
 * There is no separate `PortKind` axis anymore: `transportOf()` derives push-vs-pull
 * from the resolved type alone. `unknown` is the wildcard escape hatch and the surface
 * for the former `'Any'` tag.
 *
 * This module is dependency-free at runtime (it only imports the `SignalTypeName` *type*
 * from `signal.ts`, which is erased), so both the backend engine and the frontend editor
 * import it without pulling in node classes.
 */

import type { SignalTypeName } from './signal.js';

// ──────────────────────────────────────────────────────────────────────────────
// ResolvedType — the structural type AST
// ──────────────────────────────────────────────────────────────────────────────

export type ResolvedType =
  /** A leaf data type, pulled as a single value (e.g. Float, String, BoneRotations). */
  | { kind: 'primitive'; name: SignalTypeName }
  /** A record of named fields (the payload shape produced by `pack_event`). */
  | { kind: 'record'; fields: Record<string, ResolvedType> }
  /** An `Event<T>` push payload. Transport = push. */
  | { kind: 'event'; payload: ResolvedType }
  /** A `List<T>` pull fan-in. Transport = list. Accepts `T` or `List<T>` per source. */
  | { kind: 'list'; element: ResolvedType }
  /** Wildcard — compatible with anything, both directions. The former `'Any'`. */
  | { kind: 'unknown' };

/** Transport derived from a resolved type's outermost constructor. */
export type Transport = 'event' | 'value' | 'list';

// ──────────────────────────────────────────────────────────────────────────────
// Constructors — terse helpers for building ResolvedTypes
// ──────────────────────────────────────────────────────────────────────────────

export const RT = {
  primitive: (name: SignalTypeName): ResolvedType => ({
    kind: 'primitive',
    name,
  }),
  record: (fields: Record<string, ResolvedType>): ResolvedType => ({
    kind: 'record',
    fields,
  }),
  event: (payload: ResolvedType): ResolvedType => ({ kind: 'event', payload }),
  list: (element: ResolvedType): ResolvedType => ({ kind: 'list', element }),
  unknown: (): ResolvedType => ({ kind: 'unknown' }),
} as const;

// ──────────────────────────────────────────────────────────────────────────────
// transportOf — push / pull / fan-in, derived from the type (no separate axis)
// ──────────────────────────────────────────────────────────────────────────────

export function transportOf(t: ResolvedType): Transport {
  if (t.kind === 'event') return 'event';
  if (t.kind === 'list') return 'list';
  return 'value';
}

// ──────────────────────────────────────────────────────────────────────────────
// isAssignable — can a value of type `from` flow into a port expecting `to`?
//
// Pure structural subtyping, PLUS one documented asymmetric special case for list
// fan-in. `from` is the source (upstream output), `to` is the target (downstream input).
//
// Rules:
//   - unknown ↔ anything ............ compatible (wildcard, both directions)
//   - to.list ...................... fan-in: accepts source E *or* List<E>
//                                     (the one place transport leaks into assignability)
//   - primitive ↔ primitive ........ equal names
//   - event ↔ event ................ payloads assignable
//   - record ↔ record .............. WIDTH subtyping: every field `to` wants must exist
//                                     in `from` and be assignable (a source emitting
//                                     {a,b,c} satisfies a target wanting {a,b})
//   - mixed constructors ........... incompatible
// ──────────────────────────────────────────────────────────────────────────────

export function isAssignable(from: ResolvedType, to: ResolvedType): boolean {
  // Wildcard escape hatch, both directions.
  if (from.kind === 'unknown' || to.kind === 'unknown') return true;

  // List fan-in (asymmetric): a List<E> target accepts a source of E or List<E>.
  // This is the single transport-flavoured rule that lives inside assignability —
  // it mirrors the old value→list "many-to-one" connection allowance.
  if (to.kind === 'list') {
    if (from.kind === 'list') return isAssignable(from.element, to.element);
    return isAssignable(from, to.element);
  }

  if (from.kind !== to.kind) return false;

  switch (from.kind) {
    case 'primitive': {
      const toName = (to as Extract<ResolvedType, { kind: 'primitive' }>).name;
      if (from.name === toName) return true;
      // Entity widening (asymmetric, like list fan-in above): a concrete scene
      // entity reference flows into the generic `SceneEntity` slot, but not the
      // reverse, and `SceneNode`/`ComposeLayer` stay mutually incompatible.
      return (
        toName === 'SceneEntity' &&
        (from.name === 'SceneNode' || from.name === 'ComposeLayer')
      );
    }
    case 'event':
      return isAssignable(
        from.payload,
        (to as Extract<ResolvedType, { kind: 'event' }>).payload
      );
    case 'record': {
      const toRec = to as Extract<ResolvedType, { kind: 'record' }>;
      for (const [name, want] of Object.entries(toRec.fields)) {
        const have = from.fields[name];
        if (!have || !isAssignable(have, want)) return false;
      }
      return true;
    }
    // `from.list` is unreachable here: a list target is handled by the `to.kind ===
    // 'list'` branch above, and a non-list target with a list source is rejected by
    // the kind-mismatch guard. TS narrows `from.kind` to exclude 'list', so the
    // switch is exhaustive over primitive | event | record.
  }
  return false;
}

// ──────────────────────────────────────────────────────────────────────────────
// Human-readable rendering — for tooltips, rejection reasons, debug
// ──────────────────────────────────────────────────────────────────────────────

export function describeResolvedType(t: ResolvedType): string {
  switch (t.kind) {
    case 'primitive':
      return t.name;
    case 'event':
      return `Event<${describeResolvedType(t.payload)}>`;
    case 'list':
      return `List<${describeResolvedType(t.element)}>`;
    case 'unknown':
      return 'any';
    case 'record': {
      const inner = Object.entries(t.fields)
        .map(([k, v]) => `${k}: ${describeResolvedType(v)}`)
        .join(', ');
      return `{ ${inner} }`;
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Resolved port — a port carrying its resolved structural type
// ──────────────────────────────────────────────────────────────────────────────

export interface ResolvedPort {
  name: string;
  type: ResolvedType;
}

// ──────────────────────────────────────────────────────────────────────────────
// typeTagToResolved — lift a declared (typeTag, transport) into a ResolvedType
//
// The `'Any'` tag becomes `{ kind: 'unknown' }` (backwards-compatible surface for
// the old wildcard). Transport wraps the leaf: event → Event<leaf>, list → List<leaf>.
// ──────────────────────────────────────────────────────────────────────────────

export function typeTagToResolved(
  typeTag: SignalTypeName,
  transport: Transport
): ResolvedType {
  // `Any` and `BehaviorConfig` are both wildcard escape hatches → `unknown`.
  // BehaviorConfig is the raw-config output of `behavior_config`, historically
  // compatible with every typed value port (it carries an arbitrary config field);
  // mapping it to `unknown` preserves that wildcard behaviour under structural typing.
  const leaf: ResolvedType =
    typeTag === 'Any' || typeTag === 'BehaviorConfig'
      ? RT.unknown()
      : RT.primitive(typeTag);
  if (transport === 'event') return RT.event(leaf);
  if (transport === 'list') return RT.list(leaf);
  return leaf;
}
