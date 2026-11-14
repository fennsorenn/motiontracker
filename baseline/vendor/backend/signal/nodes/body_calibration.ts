import { SignalNode, NormalizedPose, Quaternion } from '@vspark/shared/signal';
import type { VRMBoneName } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { valueIn, valueOut, eventIn } from '@vspark/shared/node_decorators';

interface CalibrationState {
  bodyOffsets: Record<string, [number, number, number, number]>;
}

const EMPTY_STATE: CalibrationState = { bodyOffsets: {} };

export interface BodyCalibConfig {
  /**
   * Optional whitelist of VRM bone names this node captures and corrects.
   * When set, capture only stores offsets for listed bones, and apply only
   * corrects those bones — others pass through unchanged.
   * When absent, all bones are captured and corrected.
   */
  boneFilter?: readonly string[];
  /**
   * Optional pairs of [leftBone, rightBone] for symmetric body parts. On capture, if only
   * one side has a valid quaternion (e.g. the user could only present one hand because the
   * other was needed to click the button), the missing side is filled with the X-axis-mirrored
   * quaternion from the captured side. Both sides captured → both kept as-is.
   */
  mirrorPairs?: readonly (readonly [string, string])[];
  /**
   * Bones whose orientation must not follow a correction applied to their parents. For each
   * entry, the child is re-expressed so that `parents… · child` keeps the rotation it had before
   * calibration (parents are listed root-first and are bones this node may correct).
   *
   * Used where a converter already cancels a parent's measured rotation in the child — e.g. the
   * MediaPipe upper arm is solved relative to the measured shoulder shrug, so removing the shrug's
   * neutral offset must not drag the arm with it.
   */
  preserveChildren?: readonly {
    bone: string;
    parents: readonly string[];
  }[];
}

/**
 * Removes neutral-pose bias from a configurable set of bones.
 *
 * Triggers:
 *   capture — snapshot the current incoming pose as the neutral reference
 *   reset   — clear all offsets, pass through pose unmodified
 *
 * Correction formula: q_out = offset⁻¹ × q_in. The corrected pose is a PULL output
 * (`pose`) computed from the current input + stored offsets; capture/reset are event
 * handlers that mutate the stored offsets.
 *
 * Multiple instances can be chained in the graph for independent per-region
 * calibration (e.g. head chain, left arm, right arm) by setting boneFilter.
 */
@SignalNode({
  label: 'Body Calibration',
  description:
    'Removes neutral-pose bias. Use capture/reset triggers to calibrate.',
  tags: ["mocap"],
  color: '#4a5a9f',
})
export class BodyCalibration extends Node {
  static readonly kind = 'body_calibration';
  /** Captured offsets are user-produced calibration — they survive a restart.
   *  State is plain `[x,y,z,w]` arrays, so the JSON round-trip is lossless. */
  static readonly persistState = true;

  @valueIn('pose', 'NormalizedPose') poseIn!: () => NormalizedPose | undefined;
  @valueIn('mirrorSource', 'String') mirrorSource!: () =>
    | string
    | null
    | undefined;

  @valueOut('pose', 'NormalizedPose')
  pose = (): NormalizedPose | undefined => {
    const pose = this.poseIn();
    if (!pose) return undefined;
    const { bodyOffsets } = this.getState<CalibrationState>() ?? EMPTY_STATE;
    if (!bodyOffsets || Object.keys(bodyOffsets).length === 0) return pose;

    const corrected = pose.map((q, bone: VRMBoneName) => {
      const raw = bodyOffsets[bone as string];
      if (!raw) return q;
      const offset = Quaternion.fromArray(
        raw as [number, number, number, number]
      );
      return offset.isValid ? offset.invert().multiply(q) : q;
    });

    const { preserveChildren } = this.config as BodyCalibConfig;
    if (!preserveChildren?.length) return corrected;
    let out = corrected;
    for (const { bone, parents } of preserveChildren) {
      const child = corrected.get(bone as VRMBoneName);
      if (!child) continue;
      // child' = (Π parents_out)⁻¹ · (Π parents_in) · child, so Π parents_out · child' is unchanged.
      let before = Quaternion.IDENTITY;
      let after = Quaternion.IDENTITY;
      for (const p of parents) {
        before = before.multiply(pose.get(p as VRMBoneName) ?? Quaternion.IDENTITY);
        after = after.multiply(
          corrected.get(p as VRMBoneName) ?? Quaternion.IDENTITY
        );
      }
      out = out.with(
        bone as VRMBoneName,
        after.invert().multiply(before).multiply(child).normalize()
      );
    }
    return out;
  };

  @eventIn('capture', 'Trigger')
  onCapture(): void {
    const config = this.config as BodyCalibConfig;
    const filter = config.boneFilter ? new Set(config.boneFilter) : null;
    const pose = this.poseIn();
    if (!pose) return;

    const bodyOffsets: Record<string, [number, number, number, number]> = {};
    for (const [bone, q] of pose.entries()) {
      if (filter && !filter.has(bone as string)) continue;
      if (!q.isValid) continue;
      bodyOffsets[bone as string] = q.toArray();
    }
    // Symmetric fill across left/right pairs. X-axis mirror for unit quaternions:
    // (x, y, z, w) → (x, -y, -z, w).
    if (config.mirrorPairs) {
      const src = this.mirrorSource() as 'left' | 'right' | null | undefined;
      let mirrored = 0;
      for (const [a, b] of config.mirrorPairs) {
        const leftName = a;
        const rightName = b;
        const hasL = leftName in bodyOffsets;
        const hasR = rightName in bodyOffsets;

        if (src === 'left' && hasL) {
          const [x, y, z, w] = bodyOffsets[leftName];
          bodyOffsets[rightName] = [x, -y, -z, w];
          mirrored++;
        } else if (src === 'right' && hasR) {
          const [x, y, z, w] = bodyOffsets[rightName];
          bodyOffsets[leftName] = [x, -y, -z, w];
          mirrored++;
        } else if (hasL && !hasR) {
          const [x, y, z, w] = bodyOffsets[leftName];
          bodyOffsets[rightName] = [x, -y, -z, w];
          mirrored++;
        } else if (hasR && !hasL) {
          const [x, y, z, w] = bodyOffsets[rightName];
          bodyOffsets[leftName] = [x, -y, -z, w];
          mirrored++;
        }
      }
      if (mirrored > 0)
        console.log(
          `[BodyCalibration] Mirrored ${mirrored} offsets (source=${src ?? 'auto'})`
        );
    }
    this.setState({ bodyOffsets } satisfies CalibrationState);
    console.log(
      `[BodyCalibration] Captured ${Object.keys(bodyOffsets).length} bone offsets`
    );
  }

  @eventIn('reset', 'Trigger')
  onReset(): void {
    this.setState(EMPTY_STATE);
  }
}
