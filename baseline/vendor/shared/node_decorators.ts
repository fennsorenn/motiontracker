/**
 * Port decorators for the class-instance node model.
 *
 * Each decorator records port metadata into the class's shared `ctx.metadata` buffer
 * at definition time (harvested by `@SignalNode`), so ports are introspectable without
 * an instance. Field decorators return a generic identity initializer — required by
 * `tsc --strict` (TS1270) — and the engine overwrites the field with the real
 * emitter/thunk in `Node.bind()`.
 *
 *   @eventIn(name, tag)   on a METHOD  — push input; the method body is the reaction.
 *   @valueIn(name, tag)   on a FIELD   — pull input; engine assigns a `() => T` thunk.
 *   @listIn(name, tag)    on a FIELD   — pull fan-in; engine assigns a `() => T[]` thunk.
 *   @eventOut(name, tag)  on a FIELD   — push output; engine assigns an `Emitter<T>`.
 *   @valueOut(name, tag)  on a FIELD   — pull output; node sets the field to a `() => T`.
 *
 * `tag` is the leaf data-type (a `SignalTypeName`); transport comes from the decorator
 * itself. Use `'Any'` (→ `unknown`) for ports whose real type is supplied by `inferPorts`.
 */

import type { SignalTypeName } from './signal.js';
import { bufferPort } from './node.js';

// ── event input: decorate a METHOD ───────────────────────────────────────────

export function eventIn(name: string, typeTag: SignalTypeName) {
  return function <This, Args extends unknown[], Ret>(
    method: (this: This, ...args: Args) => Ret,
    ctx: ClassMethodDecoratorContext<This>
  ): (this: This, ...args: Args) => Ret {
    bufferPort(ctx.metadata, {
      name,
      direction: 'in',
      transport: 'event',
      typeTag,
      member: String(ctx.name),
    });
    return method;
  };
}

// ── value / list inputs and value output: decorate a FIELD ────────────────────

export function valueIn(name: string, typeTag: SignalTypeName) {
  return function <This, T>(
    _initial: undefined,
    ctx: ClassFieldDecoratorContext<This, T>
  ): (this: This, initial: T) => T {
    bufferPort(ctx.metadata, {
      name,
      direction: 'in',
      transport: 'value',
      typeTag,
      member: String(ctx.name),
    });
    return (initial: T): T => initial;
  };
}

export function listIn(name: string, typeTag: SignalTypeName) {
  return function <This, T>(
    _initial: undefined,
    ctx: ClassFieldDecoratorContext<This, T>
  ): (this: This, initial: T) => T {
    bufferPort(ctx.metadata, {
      name,
      direction: 'in',
      transport: 'list',
      typeTag,
      member: String(ctx.name),
    });
    return (initial: T): T => initial;
  };
}

export function eventOut(name: string, typeTag: SignalTypeName) {
  return function <This, T>(
    _initial: undefined,
    ctx: ClassFieldDecoratorContext<This, T>
  ): (this: This, initial: T) => T {
    bufferPort(ctx.metadata, {
      name,
      direction: 'out',
      transport: 'event',
      typeTag,
      member: String(ctx.name),
    });
    return (initial: T): T => initial;
  };
}

export function valueOut(name: string, typeTag: SignalTypeName) {
  return function <This, T>(
    _initial: undefined,
    ctx: ClassFieldDecoratorContext<This, T>
  ): (this: This, initial: T) => T {
    bufferPort(ctx.metadata, {
      name,
      direction: 'out',
      transport: 'value',
      typeTag,
      member: String(ctx.name),
    });
    return (initial: T): T => initial;
  };
}
