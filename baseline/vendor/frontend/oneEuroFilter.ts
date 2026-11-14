import * as THREE from 'three';

// Smoothing coefficient for a given cutoff frequency and timestep.
function alpha(cutoff: number, dt: number): number {
  const tau = 1.0 / (2 * Math.PI * cutoff);
  return 1.0 / (1.0 + tau / dt);
}

/**
 * One Euro Filter for quaternions.
 *
 * Adapts its cutoff frequency based on angular velocity: slow / noisy motion
 * gets heavy smoothing while fast movements stay crisp.
 *
 * Parameters
 *   minCutoff  Hz  Smoothing applied when nearly still. Lower = smoother at rest.
 *   beta            Speed coefficient. Higher = less lag during fast moves.
 *   dCutoff    Hz  Cutoff for the derivative estimate (rarely needs changing).
 */
export class OneEuroFilterQuat {
  private initialized = false;
  private readonly xFilt = new THREE.Quaternion();
  private readonly _tmp = new THREE.Quaternion();
  private dxFilt = 0;

  constructor(
    readonly minCutoff = 1.0,
    readonly beta = 0.3,
    readonly dCutoff = 1.0
  ) {}

  /**
   * Feed one sample. Returns a reference to the internal filtered quaternion —
   * copy x/y/z/w before calling filter() again.
   */
  filter(x: THREE.Quaternion, dt: number): THREE.Quaternion {
    if (dt <= 0) return this.xFilt;

    if (!this.initialized) {
      this.xFilt.copy(x);
      this.initialized = true;
      return this.xFilt;
    }

    // Angular velocity (rad/s): angle between the filtered and incoming quaternion.
    const dot = Math.min(1, Math.abs(this.xFilt.dot(x)));
    const dx = (2 * Math.acos(dot)) / dt;

    // Low-pass filter the velocity estimate.
    const aD = alpha(this.dCutoff, dt);
    this.dxFilt += aD * (dx - this.dxFilt);

    // Adaptive cutoff: rises with angular speed to reduce lag.
    const cutoff = this.minCutoff + this.beta * this.dxFilt;

    // Slerp toward the incoming quaternion, ensuring shortest-arc path.
    const a = alpha(cutoff, dt);
    this._tmp.copy(x);
    if (this.xFilt.dot(this._tmp) < 0) this._tmp.set(-x.x, -x.y, -x.z, -x.w);
    this.xFilt.slerp(this._tmp, a);

    return this.xFilt;
  }

  reset(): void {
    this.initialized = false;
    this.dxFilt = 0;
  }
}

/** One filter per named bone — lazily initialised. */
export class BoneFilterBank {
  private readonly filters = new Map<string, OneEuroFilterQuat>();

  constructor(
    private readonly minCutoff = 1.0,
    private readonly beta = 0.3
  ) {}

  filter(boneName: string, x: THREE.Quaternion, dt: number): THREE.Quaternion {
    let f = this.filters.get(boneName);
    if (!f) {
      f = new OneEuroFilterQuat(this.minCutoff, this.beta);
      this.filters.set(boneName, f);
    }
    return f.filter(x, dt);
  }

  reset(): void {
    this.filters.forEach((f) => f.reset());
  }
}
