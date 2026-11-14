import { SignalNode, NormalizedPose } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { listIn, valueOut } from '@vspark/shared/node_decorators';

@SignalNode({
  label: 'Pose Merge',
  description:
    'Merges multiple partial NormalizedPoses into one. Later inputs override earlier ones for any overlapping bones. Identity quaternions (0,0,0,1) from partial poses are skipped.',
  tags: ["mocap"],
  color: '#5b7a3a',
})
export class PoseMerge extends Node {
  static readonly kind = 'pose_merge';

  @listIn('poses', 'NormalizedPose') poses!: () => NormalizedPose[];

  @valueOut('pose', 'NormalizedPose')
  pose = (): NormalizedPose => {
    const poses = this.poses();
    if (!poses?.length) return new NormalizedPose();

    const merged = new Map(poses[0].entries());
    for (let i = 1; i < poses.length; i++) {
      for (const [bone, q] of poses[i].entries()) {
        // Skip identity quaternions — they indicate "no data for this bone"
        if (
          q.magnitudeSquared > 1e-9 &&
          !(q.x === 0 && q.y === 0 && q.z === 0 && q.w === 1)
        ) {
          merged.set(bone, q);
        }
      }
    }
    return new NormalizedPose(merged);
  };
}
