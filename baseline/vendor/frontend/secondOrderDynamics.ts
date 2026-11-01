import * as THREE from 'three';

/**
 * Configurable "snappiness" for broadcast bone rotations.
 *
 * This is a second-order dynamics (spring–damper) filter applied per bone,
 * layered *after* the One Euro filter (which stays responsible for absorbing
 * jitter and uneven packet delivery). Unlike a low-pass filter — which can only
 * ever lag behind the target — a second-order system can lead and overshoot the
 * target, which is what reads as "snappy" / "follow-through" without going
 * choppy (the response stays C¹-continuous).
 *
 * Parameters (see {@link PoseDynamicsConfig}):
 *   frequency  Hz  Natural frequency — how quickly it reacts. Higher = snappier.
 *   damping        ζ. <1 overshoots (snap/bounce), 1 is critical (no overshoot),
 *                  >1 is sluggish. ~0.5–0.8 gives a lively-but-controlled feel.
 *   response       r. 0 = no anticipation, >0 reacts immediately and can lead the
 *                  target (anticipatory snap), <0 winds up before moving.
 *
 * The math follows the standard semi-implicit-Euler formulation (t3ssel8r,
 * "Giving Personality to Procedural Animations using Math"), adapted from a
 * scalar to SO(3): the spring error, target velocity and output velocity are all
 * expressed as world-frame rotation vectors (axis·angle), and the output
 * orientation is integrated through the quaternion exponential map.
 */
export interface PoseDynamicsConfig {
  enabled: boolean;
  /** Natural frequency in Hz. Higher = faster / snappier. */
  frequency: number;
  /** Damping ratio ζ. <1 overshoots, 1 critical, >1 sluggish. */
  damping: number;
  /** Response r. 0 none, >0 anticipatory lead, <0 wind-up. */
  response: number;
}

export const DEFAULT_POSE_DYNAMICS: PoseDynamicsConfig = {
  enabled: false,
  frequency: 3.0,
  damping: 0.6,
  response: 1.2,
};

const MIN_FREQUENCY = 0.05; // Hz — guard against div-by-zero / runaway gains.
/** A held target-velocity estimate is dropped once no new sample has arrived
 *  for this long (or twice the last sample interval, if longer): the input has
 *  stopped, and a stale velocity would keep pushing the output. */
const STALL_SECONDS = 0.1;
const EPS = 1e-8;

// Shared scratch for the helpers below — single-threaded, reused every call.
const _rel = new THREE.Quaternion();

/**
 * Rotation vector (axis·angle, radians) of `a * b⁻¹` — the world-frame rotation
 * that takes orientation `b` to `a`. Always the shortest arc.
 */
function relRotVec(
  out: THREE.Vector3,
  a: THREE.Quaternion,
  b: THREE.Quaternion
): THREE.Vector3 {
  _rel.copy(b).invert().premultiply(a); // _rel = a * b⁻¹
  let { x, y, z, w } = _rel;
  if (w < 0) {
    // Shortest path: flip to the near hemisphere.
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }
  const vlen = Math.sqrt(x * x + y * y + z * z);
  if (vlen < EPS) return out.set(0, 0, 0);
  const angle = 2 * Math.atan2(vlen, w); // [0, π]
  const s = angle / vlen;
  return out.set(x * s, y * s, z * s);
}

/** Quaternion exponential of the rotation vector `w·dt` (axis·angle → quat). */
function rotVecToQuat(
  out: THREE.Quaternion,
  w: THREE.Vector3,
  dt: number
): THREE.Quaternion {
  const ax = w.x * dt;
  const ay = w.y * dt;
  const az = w.z * dt;
  const angle = Math.sqrt(ax * ax + ay * ay + az * az);
  if (angle < EPS) return out.set(0, 0, 0, 1);
  const half = angle / 2;
  const s = Math.sin(half) / angle;
  return out.set(ax * s, ay * s, az * s, Math.cos(half));
}

/** Second-order dynamics filter for a single bone's rotation. */
export class SecondOrderDynamicsQuat {
  private initialized = false;
  private readonly y = new THREE.Quaternion(); // output orientation
  private readonly xPrev = new THREE.Quaternion(); // previous target
  private readonly w = new THREE.Vector3(); // output angular velocity (rad/s, world)
  /** Target angular velocity (rad/s, world), estimated per input SAMPLE and
   *  held between samples. */
  private readonly _wx = new THREE.Vector3();
  /** Seconds since the last input sample, and the interval before that one. */
  private _sinceSample = 0;
  private _lastInterval = 0;

  // scratch
  private readonly _err = new THREE.Vector3();
  private readonly _dq = new THREE.Quaternion();

  /**
   * Advance one frame toward target `x`. Returns a reference to the internal
   * output quaternion — copy x/y/z/w before calling filter() again.
   *
   * `newSample` says whether `x` carries new input this frame. The target
   * velocity (the `response` term) is estimated over the time between SAMPLES,
   * not between frames: when the input updates less often than the display
   * renders (a 30 Hz tracker on a 60 Hz bus, a 60 Hz bus on a 144 Hz screen),
   * a per-frame estimate is a spike on update frames and zero in between, and
   * with heavy damping the output jerks back and forth every frame. Callers
   * that feed a fresh sample every frame can leave it at `true`.
   */
  filter(
    x: THREE.Quaternion,
    dt: number,
    frequency: number,
    damping: number,
    response: number,
    newSample = true
  ): THREE.Quaternion {
    if (dt <= 0) return this.y;

    if (!this.initialized) {
      this.y.copy(x);
      this.xPrev.copy(x);
      this.w.set(0, 0, 0);
      this._wx.set(0, 0, 0);
      this._sinceSample = 0;
      this._lastInterval = 0;
      this.initialized = true;
      return this.y;
    }

    const f = Math.max(frequency, MIN_FREQUENCY);
    const wn = 2 * Math.PI * f;
    const k1 = damping / (Math.PI * f);
    const k2 = 1 / (wn * wn);
    const k3 = (response * damping) / wn;

    // Clamp k2 so semi-implicit Euler stays stable at large dt (low frame rates).
    const k2Stable = Math.max(k2, (dt * dt) / 2 + (dt * k1) / 2, dt * k1);

    // Target angular velocity (world frame), rad/s — re-estimated per sample,
    // held in between, dropped once the input stalls.
    this._sinceSample += dt;
    if (newSample) {
      relRotVec(this._wx, x, this.xPrev).multiplyScalar(1 / this._sinceSample);
      this.xPrev.copy(x);
      this._lastInterval = this._sinceSample;
      this._sinceSample = 0;
    } else if (
      this._sinceSample > Math.max(STALL_SECONDS, 2 * this._lastInterval)
    ) {
      this._wx.set(0, 0, 0);
    }

    // Spring error: rotation that would take the output onto the target.
    relRotVec(this._err, x, this.y);

    // w += dt · (err + k3·wx − k1·w) / k2
    this.w.x +=
      (dt * (this._err.x + k3 * this._wx.x - k1 * this.w.x)) / k2Stable;
    this.w.y +=
      (dt * (this._err.y + k3 * this._wx.y - k1 * this.w.y)) / k2Stable;
    this.w.z +=
      (dt * (this._err.z + k3 * this._wx.z - k1 * this.w.z)) / k2Stable;

    // Integrate orientation: y = exp(w·dt) · y.
    rotVecToQuat(this._dq, this.w, dt);
    this.y.copy(this._dq.multiply(this.y)).normalize();

    return this.y;
  }

  reset(): void {
    this.initialized = false;
    this.w.set(0, 0, 0);
    this._wx.set(0, 0, 0);
  }
}

/** One dynamics filter per named bone — lazily initialised. */
export class BoneDynamicsBank {
  private readonly filters = new Map<string, SecondOrderDynamicsQuat>();

  filter(
    boneName: string,
    x: THREE.Quaternion,
    dt: number,
    frequency: number,
    damping: number,
    response: number,
    newSample = true
  ): THREE.Quaternion {
    let s = this.filters.get(boneName);
    if (!s) {
      s = new SecondOrderDynamicsQuat();
      this.filters.set(boneName, s);
    }
    return s.filter(x, dt, frequency, damping, response, newSample);
  }

  reset(): void {
    this.filters.forEach((s) => s.reset());
  }
}
