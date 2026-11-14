/**
 * Finger solve only: vspark's `hand_landmarks_to_bones` node on each frame's hand world landmarks.
 * Reference for motiontracker's Python port (service/src/mtservice/hands.py).
 *
 *   tsx src/hands-cli.ts < frames.jsonl > bones.jsonl
 *
 * Input: one `{ t, leftHand?, rightHand? }` per line (21 points as [x,y,z] or {x,y,z}).
 * Output: one `{ t, bones: { <vrmFingerBone>: [x,y,z,w] } }` per line, nothing held between frames.
 */
import { createInterface } from 'readline';
import type { NormalizedPose } from '@vspark/shared/signal';
import type { NodeBindContext, Thunk } from '@vspark/shared/node';
import { HandLandmarksToBones } from '../vendor/backend/signal/nodes/hand_landmarks_to_bones.js';

type Landmark = { x: number; y: number; z: number };
const toLm = (p: unknown): Landmark =>
  Array.isArray(p) ? { x: p[0], y: p[1], z: p[2] } : (p as Landmark);

function handNode(side: 'left' | 'right', landmarks: () => Landmark[] | undefined): () => NormalizedPose | undefined {
  const outputs: Record<string, Thunk<unknown>> = {};
  const config: Record<string, unknown> = { side };
  const noop = { emit() {} };
  const ctx = {
    config,
    getState: () => undefined,
    setState: () => {},
    makeEmitter: () => noop,
    valueThunk: (port: string) => () => (port === 'landmarks' ? landmarks() : config[port]),
    listThunk: () => () => [],
    registerOutputThunk: (port: string, fn: Thunk<unknown>) => {
      outputs[port] = fn;
    },
    registerHandler: () => {},
    makeDynamicEmitter: () => noop,
    dynamicValueThunk: (port: string) => () => config[port],
    registerDynamicOutputs: () => {},
    isEnabled: () => true,
  } as unknown as NodeBindContext;
  new HandLandmarksToBones().bind(ctx);
  return outputs.pose as () => NormalizedPose | undefined;
}

let cur: { leftHand?: Landmark[]; rightHand?: Landmark[] } = {};
const left = handNode('left', () => cur.leftHand);
const right = handNode('right', () => cur.rightHand);

for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const f = JSON.parse(line);
  cur = {
    leftHand: f.leftHand?.length ? f.leftHand.map(toLm) : undefined,
    rightHand: f.rightHand?.length ? f.rightHand.map(toLm) : undefined,
  };
  const bones = { ...(left()?.toRecord() ?? {}), ...(right()?.toRecord() ?? {}) };
  process.stdout.write(JSON.stringify({ t: f.t, bones }) + '\n');
}
