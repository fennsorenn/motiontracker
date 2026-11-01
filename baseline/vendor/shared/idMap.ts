/**
 * Id-keyed collections inside a synced document.
 *
 * A document field holding a LIST of things — a clip's lanes, a lane's
 * keyframes, a graph's nodes — cannot be an array on the mesh. An array is one
 * value at one path, so any two peers touching different elements write the
 * same path and last-writer-wins throws one edit away: A drags keyframe 3 while
 * B drags keyframe 7, and whoever lands second wins the whole lane. Keyed by
 * id, each element is its own path (`lanes.<laneId>.keyframes.<kfId>.value`),
 * and those two edits merge because they never collide.
 *
 * Order is NOT storage. A keyed map has none, which is the point: order that
 * matters must be derivable from the elements themselves (keyframes and events
 * sort by `t`) or carried in a field (a fractional index, see fracIndex.ts).
 * An element's position is then a value two peers can converge on, rather than
 * an accident of array index.
 *
 * ## Deletion
 *
 * `setPath` can write a key but not remove one, so a deleted element is present
 * with a `null` value — the ordinary LWW tombstone. Readers must skip nulls,
 * which is what {@link itemsOf} is for; treating the raw map as `Record<string,
 * T>` would hand a deleted element back as `null` and blow up downstream.
 *
 * Tombstones do not accumulate on disk: persistence writes rows for the live
 * elements only, so a reload rebuilds the map without them. They live in the
 * replica for the session, which is exactly long enough for a peer that missed
 * the delete to receive it.
 */

/** A document field holding elements keyed by id. `null` = deleted. */
export type IdMap<T> = Record<string, T | null>;

/** The live elements, in no particular order — sort the result by whatever
 *  field carries order for that type. Skips tombstones. */
export function itemsOf<T>(map: IdMap<T> | undefined | null): T[] {
  if (!map) return [];
  const out: T[] = [];
  for (const v of Object.values(map)) if (v != null) out.push(v);
  return out;
}

/** One live element, or undefined when absent or deleted. */
export function itemOf<T>(
  map: IdMap<T> | undefined | null,
  id: string
): T | undefined {
  return map?.[id] ?? undefined;
}

/** Build a map from elements that carry their own id. For converting at a
 *  boundary — a REST body, a preset bundle, a DB read — never for round-tripping
 *  a map you already have (that would drop its tombstones). */
export function byId<T extends { id: string }>(items: readonly T[]): IdMap<T> {
  const out: IdMap<T> = {};
  for (const it of items) out[it.id] = it;
  return out;
}

/** Live elements sorted by a numeric field — the common case for timed
 *  elements (`t`), where the order is a property of the data, not of storage. */
export function sortedBy<T>(
  map: IdMap<T> | undefined | null,
  key: (item: T) => number
): T[] {
  return itemsOf(map).sort((a, b) => key(a) - key(b));
}
