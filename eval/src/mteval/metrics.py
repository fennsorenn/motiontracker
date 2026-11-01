"""Metrics. All are time-based (degrees per second, milliseconds), so outputs at different rates
(10 Hz solver, 60 Hz rendered) are comparable.

Label-free (real recordings):
  speed_percentiles   p50/p90/p99 of bone angular speed, overall and per segment label.
  rest_jitter         RMS angular speed during still segments (deg/s). Lower = calmer at rest.
  rest_noise          RMS angle to the local mean rotation (±250 ms) during still segments (deg).
  twitch_rate         events per minute where angular speed exceeds a threshold during still
                      segments (a run of consecutive fast intervals counts once).
  dropout_jump        peak angular speed within 500 ms after a hand appears or disappears in the
                      input, averaged over transitions (deg/s), for that side's arm/hand bones.
  presence            fraction of output frames in which the bone is present.
  lag_vs_detector     delay of the output relative to the detector: cross-correlation of an input
                      landmark angular-speed signal with the matching bone's angular speed. Measures
                      what the solver and smoothing add on top of the detector, not camera latency.

Labeled (need ground truth; ready for synthetic M1 data):
  rotation_error      geodesic error vs ground truth (deg): mean, median, p95.
  onset_lag           cross-correlation lag of predicted vs ground-truth angular speed (ms).
  peak_velocity_ratio predicted / ground-truth peak angular speed per movement, median.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quat import angle_deg, local_mean
from .session import (
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseTrack,
    Segment,
    Session,
)

FINGER_PARTS = ("Thumb", "Index", "Middle", "Ring", "Little")


def bone_group(name: str) -> str:
    if any(p in name for p in FINGER_PARTS):
        return "fingers"
    if name in ("leftHand", "rightHand"):
        return "hands"
    if "UpperArm" in name or "LowerArm" in name:
        return "arms"
    if name in ("leftShoulder", "rightShoulder"):
        return "shoulders"
    return "torso_head"


# ── angular speed ────────────────────────────────────────────────────────────────


@dataclass
class SpeedSeries:
    """Angular speed between consecutive present frames. `t` is the interval midpoint (ms)."""

    t: np.ndarray
    speed: np.ndarray  # deg/s
    t_start: np.ndarray
    t_end: np.ndarray


def angular_speed(times: np.ndarray, q: np.ndarray, max_gap_ms: float | None = None) -> SpeedSeries:
    """Angular speed of one bone track. Intervals spanning a gap (missing frames or a time step
    above `max_gap_ms`, default 3x the median step) are skipped."""
    ok = ~np.isnan(q[:, 0])
    t, qq = times[ok], q[ok]
    if len(t) < 2:
        e = np.array([])
        return SpeedSeries(e, e, e, e)
    dt = np.diff(t)
    if max_gap_ms is None:
        max_gap_ms = 3.0 * float(np.median(np.diff(times))) if len(times) > 1 else np.inf
    # Also break where frames were missing in between (index jump in the full track).
    idx = np.flatnonzero(ok)
    contiguous = np.diff(idx) == 1
    keep = (dt > 0) & (dt <= max_gap_ms) & contiguous
    ang = angle_deg(qq[1:], qq[:-1])
    speed = ang / (dt / 1000.0)
    return SpeedSeries(
        t=((t[1:] + t[:-1]) / 2)[keep],
        speed=speed[keep],
        t_start=t[:-1][keep],
        t_end=t[1:][keep],
    )


def _mask(t: np.ndarray, segments: list[Segment]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for s in segments:
        m |= s.contains(t)
    return m


def _pct(x: np.ndarray) -> dict[str, float] | None:
    if len(x) == 0:
        return None
    p50, p90, p99 = np.percentile(x, [50, 90, 99])
    return {"p50": float(p50), "p90": float(p90), "p99": float(p99), "n": int(len(x))}


# ── label-free metrics ───────────────────────────────────────────────────────────


def speed_percentiles(track: PoseTrack, segments: list[Segment]) -> dict[str, dict]:
    labels = sorted({s.label for s in segments})
    out: dict[str, dict] = {}
    for bone, q in track.bones.items():
        sp = angular_speed(track.times, q)
        entry = {"all": _pct(sp.speed)}
        for label in labels:
            entry[label] = _pct(sp.speed[_mask(sp.t, [s for s in segments if s.label == label])])
        out[bone] = entry
    return out


def rest_jitter(track: PoseTrack, still: list[Segment]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for bone, q in track.bones.items():
        sp = angular_speed(track.times, q)
        s = sp.speed[_mask(sp.t, still)]
        out[bone] = float(np.sqrt(np.mean(s**2))) if len(s) else None
    return out


def rest_noise(track: PoseTrack, still: list[Segment], half_window_ms: float = 250.0) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    t = track.times
    in_still = _mask(t, still)
    for bone, q in track.bones.items():
        errs = []
        for k in np.flatnonzero(in_still & ~np.isnan(q[:, 0])):
            lo = int(np.searchsorted(t, t[k] - half_window_ms))
            hi = int(np.searchsorted(t, t[k] + half_window_ms, side="right"))
            window_ok = ~np.isnan(q[lo:hi, 0])
            if window_ok.sum() < 3:
                continue
            sub = q[lo:hi][window_ok]
            center = int(np.argmin(np.abs(t[lo:hi][window_ok] - t[k])))
            errs.append(float(angle_deg(q[k], local_mean(sub, center, 0, len(sub)))))
        out[bone] = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    return out


def twitch_rate(
    track: PoseTrack, still: list[Segment], threshold_deg_s: float = 180.0
) -> dict[str, float | None]:
    minutes = sum(s.t1 - s.t0 for s in still) / 60000.0
    out: dict[str, float | None] = {}
    for bone, q in track.bones.items():
        if minutes <= 0:
            out[bone] = None
            continue
        sp = angular_speed(track.times, q)
        m = _mask(sp.t, still)
        fast = (sp.speed > threshold_deg_s) & m
        # Count runs: a fast interval whose predecessor (if adjacent) was not fast.
        events = 0
        for k in np.flatnonzero(fast):
            adjacent = k > 0 and sp.t_end[k - 1] == sp.t_start[k]
            if not (adjacent and fast[k - 1]):
                events += 1
        out[bone] = events / minutes
    return out


def presence(track: PoseTrack) -> dict[str, float]:
    n = len(track.times)
    return {b: float(np.mean(~np.isnan(q[:, 0]))) if n else 0.0 for b, q in track.bones.items()}


def dropout_jump(track: PoseTrack, session: Session, window_ms: float = 500.0) -> dict[str, dict]:
    """For each side, find input frames where the hand stream appears/disappears, then take the
    peak angular speed of that side's arm, hand and finger bones within `window_ms` after it."""
    t_in = session.times
    out: dict[str, dict] = {}
    for side in ("left", "right"):
        present = session.present(f"{side}Hand")
        trans = t_in[1:][present[1:] != present[:-1]]
        bones = [b for b in track.bones if b.startswith(side) and bone_group(b) in ("arms", "hands", "fingers")]
        for bone in bones:
            sp = angular_speed(track.times, track.bones[bone])
            peaks = []
            for tt in trans:
                # Include the interval that ends on the transition frame (the snap itself).
                m = (sp.t_end >= tt) & (sp.t_start < tt + window_ms)
                if m.any():
                    peaks.append(float(sp.speed[m].max()))
            out[bone] = {"mean_peak": float(np.mean(peaks)) if peaks else None, "transitions": len(trans)}
    return out


def _direction_speed(t: np.ndarray, vecs: list[np.ndarray | None]) -> SpeedSeries:
    """Angular speed (deg/s) of a per-frame 3D direction vector."""
    ts, sp = [], []
    for k in range(1, len(t)):
        a, b = vecs[k - 1], vecs[k]
        dt = (t[k] - t[k - 1]) / 1000.0
        if a is None or b is None or dt <= 0:
            continue
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ts.append((t[k] + t[k - 1]) / 2)
        sp.append(np.degrees(np.arccos(np.clip(cos, -1, 1))) / dt)
    ts_a, sp_a = np.array(ts), np.array(sp)
    return SpeedSeries(ts_a, sp_a, ts_a, ts_a)


def xcorr_lag(
    ref: SpeedSeries,
    other: SpeedSeries,
    hold_other: bool = False,
    max_lag_ms: float = 1000.0,
    min_lag_ms: float = -200.0,
    grid_hz: float = 100.0,
) -> dict[str, float | None]:
    """Lag (ms) at which `other` best matches `ref` (positive = other is late), plus the peak
    normalized correlation. Both signals are resampled linearly to a uniform grid (speeds are
    stamped at interval midpoints, so this is unbiased across sample rates); `hold_other` uses a
    zero-order hold for `other` instead, which adds about half its frame interval."""
    if len(ref.t) < 10 or len(other.t) < 10:
        return {"lag_ms": None, "corr": None}
    t0, t1 = max(ref.t[0], other.t[0]), min(ref.t[-1], other.t[-1])
    if t1 - t0 < 2000:
        return {"lag_ms": None, "corr": None}
    step = 1000.0 / grid_hz
    grid = np.arange(t0, t1, step)
    a = np.interp(grid, ref.t, ref.speed)
    if hold_other:
        idx = np.clip(np.searchsorted(other.t, grid, side="right") - 1, 0, len(other.t) - 1)
        b = other.speed[idx]
    else:
        b = np.interp(grid, other.t, other.speed)
    a = (a - a.mean()) / (a.std() or 1.0)
    b = (b - b.mean()) / (b.std() or 1.0)
    best_lag, best_c = None, -np.inf
    for lag in np.arange(int(min_lag_ms / step), int(max_lag_ms / step) + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        if len(x) < 50:
            continue
        c = float(np.mean(x * y))
        if c > best_c:
            best_c, best_lag = c, lag * step
    return {"lag_ms": best_lag, "corr": best_c if best_lag is not None else None}


def lag_vs_detector(track: PoseTrack, session: Session) -> dict[str, dict]:
    """Input signals from pose world landmarks: upper-arm direction (shoulder→elbow) for the upper
    arms, forearm direction (elbow→wrist) for the lower arms, head direction (ear midpoint→nose)
    for the neck. Landmark sides map to the same-named avatar bones (vspark mirrors the frame)."""
    t = session.times
    pose = session.stream("pose")

    def vec(i: int, j: int, k: int | None = None) -> list[np.ndarray | None]:
        out = []
        for p in pose:
            if p is None:
                out.append(None)
            elif k is None:
                out.append(p[j, :3] - p[i, :3])
            else:
                out.append(p[k, :3] - (p[i, :3] + p[j, :3]) / 2)
        return out

    pairs = {
        "leftUpperArm": vec(LEFT_SHOULDER, LEFT_ELBOW),
        "rightUpperArm": vec(RIGHT_SHOULDER, RIGHT_ELBOW),
        "leftLowerArm": vec(LEFT_ELBOW, LEFT_WRIST),
        "rightLowerArm": vec(RIGHT_ELBOW, RIGHT_WRIST),
        "neck": vec(LEFT_EAR, RIGHT_EAR, NOSE),
    }
    out: dict[str, dict] = {}
    for bone, vecs in pairs.items():
        if bone not in track.bones:
            continue
        out[bone] = xcorr_lag(_direction_speed(t, vecs), angular_speed(track.times, track.bones[bone]))
    return out


# ── labeled metrics ──────────────────────────────────────────────────────────────


def _hold_sample(track_t: np.ndarray, q: np.ndarray, at: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(track_t, at, side="right") - 1
    res = np.full((len(at), 4), np.nan)
    ok = idx >= 0
    res[ok] = q[idx[ok]]
    return res


def rotation_error(pred: PoseTrack, gt: PoseTrack) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {}
    for bone, g in gt.bones.items():
        if bone not in pred.bones:
            out[bone] = None
            continue
        p = _hold_sample(pred.times, pred.bones[bone], gt.times)
        ok = ~np.isnan(p[:, 0]) & ~np.isnan(g[:, 0])
        if not ok.any():
            out[bone] = None
            continue
        e = angle_deg(p[ok], g[ok])
        out[bone] = {
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)),
            "coverage": float(ok.mean()),
        }
    return out


def onset_lag(pred: PoseTrack, gt: PoseTrack) -> dict[str, dict]:
    return {
        b: xcorr_lag(angular_speed(gt.times, g), angular_speed(pred.times, pred.bones[b]))
        for b, g in gt.bones.items()
        if b in pred.bones
    }


def peak_velocity_ratio(
    pred: PoseTrack, gt: PoseTrack, min_peak_deg_s: float = 60.0, window_ms: float = 300.0
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for bone, g in gt.bones.items():
        if bone not in pred.bones:
            continue
        gs = angular_speed(gt.times, g)
        ps = angular_speed(pred.times, pred.bones[bone])
        ratios = []
        last_peak = -np.inf
        for k in range(1, len(gs.speed) - 1):
            s = gs.speed[k]
            if s < min_peak_deg_s or s < gs.speed[k - 1] or s < gs.speed[k + 1]:
                continue
            if gs.t[k] - last_peak < window_ms:
                continue
            last_peak = gs.t[k]
            m = (ps.t >= gs.t[k] - window_ms) & (ps.t <= gs.t[k] + window_ms)
            if m.any():
                ratios.append(float(ps.speed[m].max() / s))
        out[bone] = {"median": float(np.median(ratios)) if ratios else None, "peaks": len(ratios)}
    return out
