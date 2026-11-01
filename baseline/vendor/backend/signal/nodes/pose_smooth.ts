import { SignalNode, NormalizedPose, Quaternion } from '@vspark/shared/signal';
import type { VRMBoneName, BoneConfidence } from '@vspark/shared/signal';
import { Node } from '@vspark/shared/node';
import { valueIn, valueOut } from '@vspark/shared/node_decorators';

/** Lag (seconds) at confidence 0; it falls off quadratically to none at confidence 1. */
const DEFAULT_MAX_LAG = 0.6;
/** Time constant (seconds) for a bone's confidence to climb back after a drop. */
const DEFAULT_RISE_TIME = 0.3;
/** A bone missing for longer than this starts from scratch when it returns. */
const FORGET_MS = 1000;
/** Jump hold: a rotation change above this many degrees in one frame … */
const DEFAULT_HOLD_DEG = 60;
/** … is held until the new reading has persisted this many frames. */
const DEFAULT_HOLD_FRAMES = 3;

const clamp = (x: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, x));

const angleDeg = (a: Quaternion, b: Quaternion): number =>
  (2 *
    Math.acos(
      Math.min(1, Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w))
    ) *
    180) /
  Math.PI;

interface BoneTrack {
  q: Quaternion;
  conf: number;
  seenAt: number;
  /** Last accepted (not held) reading, which jumps are measured against. */
  ref: Quaternion;
  /** Consecutive frames the current jump has been held. */
  held: number;
}

/**
 * Confidence-weighted pose smoothing. Each bone eases toward its incoming rotation with a lag
 * that grows as its confidence drops — a fully confident bone passes straight through (the
 * frontend applies its own jitter filter), an unreliable one (e.g. a landmark MediaPipe can only
 * estimate) drifts slowly instead of jumping.
 *
 * Confidence drops take effect at once, but climb back over `riseTime`, so a bone whose tracking
 * returns eases into the new reading instead of snapping to it. A bone that reappears within
 * `FORGET_MS` resumes from where it was, at confidence 0.
 *
 * Jump hold (`jumpHold`): a reading that turns a bone by more than `holdDeg` in one frame is held
 * back until it has persisted `holdFrames` frames. Trackers that alternate between two readings
 * of the same pose (MediaPipe's hand model can, with the fingers toward the camera) then never get
 * through; a real fast move arrives that many frames late.
 */
@SignalNode({
  label: 'Pose Smooth',
  description:
    'Eases each bone toward its tracked rotation with a lag that grows as its confidence (0–1) drops. Confident bones pass through; unreliable ones drift instead of jumping. maxLag = lag in seconds at confidence 0, riseTime = how long confidence takes to recover after a drop. jumpHold holds back single-frame jumps above holdDeg until they persist holdFrames frames.',
  tags: ['mocap'],
  color: '#4a7a6a',
})
export class PoseSmooth extends Node {
  static readonly kind = 'pose_smooth';

  @valueIn('pose', 'NormalizedPose') poseIn!: () => NormalizedPose | undefined;
  @valueIn('confidence', 'BoneConfidence') confidenceIn!: () =>
    | BoneConfidence
    | undefined;
  @valueIn('maxLag', 'Float') maxLagIn!: () => number | undefined;
  @valueIn('riseTime', 'Float') riseTimeIn!: () => number | undefined;
  /** false = pass the pose through untouched. */
  @valueIn('enabled', 'Bool') enabledIn!: () => boolean | null | undefined;
  @valueIn('jumpHold', 'Bool') jumpHoldIn!: () => boolean | null | undefined;
  @valueIn('holdDeg', 'Float') holdDegIn!: () => number | undefined;
  @valueIn('holdFrames', 'Float') holdFramesIn!: () => number | undefined;

  private readonly _tracks = new Map<VRMBoneName, BoneTrack>();
  private _lastAt = 0;
  /** Memo so multiple pulls in one frame integrate exactly once. */
  private _memoFor: NormalizedPose | null = null;
  private _memo: NormalizedPose | null = null;

  @valueOut('pose', 'NormalizedPose')
  pose = (): NormalizedPose | undefined => {
    const pose = this.poseIn();
    if (!pose) return undefined;
    if (!(this.enabledIn() ?? true)) {
      this._tracks.clear();
      this._lastAt = 0;
      return pose;
    }
    if (this._memoFor === pose && this._memo) return this._memo;

    const confidence = this.confidenceIn() ?? {};
    const maxLag = Math.max(0, this.maxLagIn() ?? DEFAULT_MAX_LAG);
    const riseTime = Math.max(0, this.riseTimeIn() ?? DEFAULT_RISE_TIME);
    const holdDeg =
      (this.jumpHoldIn() ?? true) ? (this.holdDegIn() ?? DEFAULT_HOLD_DEG) : 0;
    const holdFrames = Math.max(1, this.holdFramesIn() ?? DEFAULT_HOLD_FRAMES);

    const now = Date.now();
    const dt =
      this._lastAt === 0 ? 0 : clamp((now - this._lastAt) / 1000, 0, 0.5);
    this._lastAt = now;

    const out = pose.map((reading, bone) => {
      const target = clamp(confidence[bone] ?? 1, 0, 1);
      const prev = this._tracks.get(bone);
      if (!prev || now - prev.seenAt > FORGET_MS) {
        this._tracks.set(bone, {
          q: reading,
          conf: target,
          seenAt: now,
          ref: reading,
          held: 0,
        });
        return reading;
      }
      // Jump hold — measured against the last accepted reading, not the smoothed output, so the
      // smoothing lag itself never counts as a jump.
      let q = reading;
      let held = 0;
      if (
        holdDeg > 0 &&
        angleDeg(reading, prev.ref) > holdDeg &&
        prev.held + 1 < holdFrames
      ) {
        q = prev.ref;
        held = prev.held + 1;
      }
      // Drops land at once; recovery is eased.
      const conf =
        target <= prev.conf || riseTime === 0
          ? target
          : prev.conf + (target - prev.conf) * (1 - Math.exp(-dt / riseTime));
      const lag = maxLag * (1 - conf) * (1 - conf);
      const next = lag <= 1e-4 ? q : prev.q.slerp(q, 1 - Math.exp(-dt / lag));
      this._tracks.set(bone, { q: next, conf, seenAt: now, ref: q, held });
      return next;
    });

    // A bone that went missing resumes at confidence 0 when it returns (if not yet forgotten).
    for (const track of this._tracks.values()) {
      if (track.seenAt !== now) track.conf = 0;
    }

    this._memoFor = pose;
    this._memo = out;
    return out;
  };
}
