import { SignalNode } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { valueIn, valueOut } from '@vspark/shared/node_decorators';

type Landmark = { x: number; y: number; z: number; visibility?: number };

// BlazePose wrist indices.
const LEFT_WRIST = 15;
const RIGHT_WRIST = 16;
const MIN_VIS = 0.5;

/**
 * Compares the vertical positions of the left and right wrist landmarks and outputs
 * which one is higher. Used to drive "use the higher hand as reference" for one-hand
 * calibration workflows.
 *
 * MediaPipe world landmarks use +Y down (image convention). We don't flip Y here because
 * we're only comparing relative positions — "higher" means smaller raw Y.
 *
 * Returns 'left', 'right', or null (when neither wrist is reliably visible).
 */
@SignalNode({
  label: 'Hand Height Compare',
  description:
    'Outputs which wrist (left/right) is currently higher in the pose landmarks. Useful for one-hand calibration where the higher hand is the user-chosen reference.',
  tags: ["mocap"],
  color: '#2a4a6a',
})
export class HandHeightCompare extends Node {
  static readonly kind = 'hand_height_compare';

  @valueIn('pose', 'LandmarkList') pose!: () => Landmark[] | undefined;

  @valueOut('side', 'String')
  side = (): string => {
    const pts = this.pose();
    if (!pts || pts.length < 17) return null as unknown as string;

    const lw = pts[LEFT_WRIST];
    const rw = pts[RIGHT_WRIST];
    const lOk = (lw.visibility ?? 1) >= MIN_VIS;
    const rOk = (rw.visibility ?? 1) >= MIN_VIS;

    if (!lOk && !rOk) return null as unknown as string;
    if (lOk && !rOk) return 'left';
    if (rOk && !lOk) return 'right';
    // Both visible — smaller raw Y means higher in image space.
    return lw.y < rw.y ? 'left' : 'right';
  };
}
