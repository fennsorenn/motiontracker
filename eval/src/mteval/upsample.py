"""Smoothing-free upsampling of a system's output to the render rate ("glide to latest").

For systems meant to replace both vspark's solver and its frontend One Euro filter: the avatar
is drawn at 60 Hz but poses arrive at the detector rate, so something must fill the frames in
between without smoothing. On each new pose, every bone slerps from what is currently shown to
the new pose, finishing one render frame before the next pose is expected (expected interval =
mean of the last 10 input intervals; causal), then holds. Finishing a frame early makes the glide
arrive even though render ticks and input times don't line up. Adds about half an input interval
of lag. Render ticks match the baseline's render stage: from the
first pose's time, every 1000/hz ms, up to the last pose.
"""

from __future__ import annotations

import numpy as np

from .quat import normalize, slerp
from .session import PoseTrack

WINDOW = 10  # input intervals averaged for the glide duration


def glide(track: PoseTrack, hz: float = 60.0) -> PoseTrack:
    times = track.times
    if len(times) == 0:
        return PoseTrack(np.zeros(0), {n: np.zeros((0, 4)) for n in track.bones})
    dt = 1000.0 / hz
    ticks = np.arange(times[0], times[-1] + 1e-9, dt)
    out = {n: np.full((len(ticks), 4), np.nan) for n in track.bones}
    shown = {n: track.bones[n][0].copy() for n in track.bones}  # NaN where absent
    start = {n: shown[n].copy() for n in track.bones}
    i, t_new, dur = 0, times[0], dt  # dur: glide duration, ms
    for k, t in enumerate(ticks):
        new = False
        while i + 1 < len(times) and times[i + 1] <= t:
            i += 1
            new = True
        if new:
            past = np.diff(times[max(0, i - WINDOW):i + 1])
            dur = max(dt, float(past.mean()) - dt)
            t_new = times[i]
            start = {n: shown[n].copy() for n in track.bones}
        u = min(1.0, max(0.0, (t - t_new) / dur))
        for n, q in track.bones.items():
            target = q[i]
            if np.isnan(target).any():
                shown[n] = target  # bone absent in the newest pose: absent in the output
            elif np.isnan(start[n]).any():
                shown[n] = normalize(target)  # bone (re)appears: snap
            else:
                shown[n] = slerp(start[n], target, u)
            out[n][k] = shown[n]
    return PoseTrack(ticks, out)
