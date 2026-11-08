/**
 * Run recorded MediaPipe Holistic frames through the baseline solver.
 *
 *   pnpm run-baseline --input frames.jsonl [--output poses.jsonl] [--stage solver|rendered]
 *                     [--render-hz 60] [--calibration calib.json] [--capture-head-at <ms>]
 *                     [--capture-fingers-at <ms>] [--no-hold-missing]
 *   pnpm run-baseline --poses poses.jsonl --stage rendered [--output out.jsonl] [--render-hz 60]
 *
 * `--poses` skips the solver: it runs a precomputed pose track (any system's output, same format as
 * this CLI's) through the frontend render stage only, so other systems get identical filtering.
 *
 * Input: one JSON object per line, `{ t, pose?, face?, leftHand?, rightHand? }`. Landmarks may be
 * objects `{x,y,z,visibility?}` or compact arrays `[x,y,z,visibility?]` (vspark fixture format).
 * Output: one `{ t, bones: { <vrmBone>: [x,y,z,w] } }` per line (VRM normalized bone rotations).
 */
import fs from 'node:fs';
import { parseArgs } from 'node:util';
import {
  BaselineSolver,
  renderStage,
  type InputFrame,
  type Landmark,
  type OutputFrame,
} from './pipeline.js';

type RawLandmark = Landmark | number[];

export function inflate(list: RawLandmark[] | null | undefined): Landmark[] | undefined {
  if (!list) return undefined;
  return list.map((p) =>
    Array.isArray(p)
      ? { x: p[0], y: p[1], z: p[2], ...(p.length > 3 ? { visibility: p[3] } : {}) }
      : p
  );
}

export function parseFrame(raw: Record<string, unknown>): InputFrame {
  return {
    t: Number(raw.t),
    pose: inflate(raw.pose as RawLandmark[] | null),
    face: inflate(raw.face as RawLandmark[] | null),
    leftHand: inflate(raw.leftHand as RawLandmark[] | null),
    rightHand: inflate(raw.rightHand as RawLandmark[] | null),
  };
}

function main(): void {
  // Vendored nodes log to console.log; keep stdout clean for JSONL output.
  console.log = console.error;
  const { values } = parseArgs({
    options: {
      input: { type: 'string' },
      poses: { type: 'string' },
      output: { type: 'string' },
      stage: { type: 'string', default: 'solver' },
      'render-hz': { type: 'string', default: '60' },
      calibration: { type: 'string' },
      'capture-head-at': { type: 'string' },
      'capture-fingers-at': { type: 'string' },
      'no-hold-missing': { type: 'boolean', default: false },
    },
  });
  if (values.stage !== 'solver' && values.stage !== 'rendered')
    throw new Error('--stage must be "solver" or "rendered"');
  if (values.poses !== undefined) {
    if (values.input !== undefined) throw new Error('--input and --poses are mutually exclusive');
    if (values.stage !== 'rendered') throw new Error('--poses needs --stage rendered');
    const poses = readPoses(values.poses);
    write(renderStage(poses, { hz: Number(values['render-hz']) }), values.output);
    return;
  }
  if (!values.input) throw new Error('--input or --poses is required');

  const solver = new BaselineSolver({
    holdMissing: !values['no-hold-missing'],
    calibration: values.calibration
      ? JSON.parse(fs.readFileSync(values.calibration, 'utf8'))
      : undefined,
  });
  const pending = {
    head: values['capture-head-at'] !== undefined ? Number(values['capture-head-at']) : undefined,
    fingers:
      values['capture-fingers-at'] !== undefined ? Number(values['capture-fingers-at']) : undefined,
  };

  const poses: OutputFrame[] = [];
  const lines = fs.readFileSync(values.input, 'utf8').split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const frame = parseFrame(JSON.parse(line));
    const pose = solver.step(frame);
    if (pose) poses.push(pose);
    // Capture after the frame is solved, on that frame's inputs (like pressing the button).
    for (const which of ['head', 'fingers'] as const) {
      const at = pending[which];
      if (at !== undefined && frame.t >= at) {
        solver.capture(which, frame.t);
        pending[which] = undefined;
      }
    }
  }

  write(values.stage === 'rendered' ? renderStage(poses, { hz: Number(values['render-hz']) }) : poses, values.output);
}

/** Pose track JSONL -> frames; only `t` and `bones` are kept (extra keys like `conf` are dropped). */
export function readPoses(path: string): OutputFrame[] {
  return fs
    .readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => {
      const raw = JSON.parse(line);
      return { t: Number(raw.t), bones: raw.bones };
    });
}

function write(frames: OutputFrame[], output?: string): void {
  const text = frames.map((p) => JSON.stringify(p)).join('\n') + '\n';
  if (output) fs.writeFileSync(output, text);
  else process.stdout.write(text);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
