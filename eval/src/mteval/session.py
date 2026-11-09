"""Recorded sessions: landmarks (+ optional video and protocol segments) and system outputs.

A session is either a directory written by the recorder (tools/recorder) containing
`session.json` and `landmarks.jsonl` (and usually `video.webm`), or a bare landmarks JSONL file.

landmarks.jsonl: one frame per line, `{t, pose?, face?, leftHand?, rightHand?}`; t in ms; landmarks
as `[x, y, z, visibility?]` arrays or `{x, y, z, visibility?}` objects. This is the format the
baseline CLI consumes.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MediaPipe pose landmark indices.
NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16


@dataclass
class Segment:
    label: str
    t0: float  # ms
    t1: float  # ms
    variant: str | None = None  # e.g. 'arms-below-frame' for still segments (seated-v2)
    settle_ms: float | None = None  # time to get into the pose; None: the metric's default

    def contains(self, t: np.ndarray) -> np.ndarray:
        return (t >= self.t0) & (t < self.t1)


@dataclass
class Session:
    name: str
    landmarks_path: Path
    frames: list[dict]
    segments: list[Segment] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def times(self) -> np.ndarray:
        return np.array([f["t"] for f in self.frames], dtype=float)

    def stream(self, key: str) -> list[np.ndarray | None]:
        """Per-frame landmark array (N, 3 or 4) for one stream, None where absent."""
        out: list[np.ndarray | None] = []
        for f in self.frames:
            v = f.get(key)
            if not v:
                out.append(None)
            elif isinstance(v[0], dict):
                out.append(np.array([[p["x"], p["y"], p["z"], p.get("visibility", np.nan)] for p in v]))
            else:
                out.append(np.array([list(p) + [np.nan] * (4 - len(p)) for p in v], dtype=float))
        return out

    def present(self, key: str) -> np.ndarray:
        return np.array([bool(f.get(key)) for f in self.frames])

    def segments_labeled(self, label: str) -> list[Segment]:
        return [s for s in self.segments if s.label == label]


def load_session(path: str | Path) -> Session:
    path = Path(path)
    if path.is_dir():
        meta = json.loads((path / "session.json").read_text())
        lm = path / meta.get("landmarks", {}).get("file", "landmarks.jsonl")
        segments = [
            Segment(s["label"], float(s["t0"]), float(s["t1"]), s.get("variant"), s.get("settleMs"))
            for s in meta.get("segments", [])
        ]
        name = path.name  # the folder name is unique (timestamped); meta["name"] is a free label
    else:
        meta, lm, segments, name = {}, path, [], path.stem
    frames = [json.loads(line) for line in lm.read_text().splitlines() if line.strip()]
    frames.sort(key=lambda f: f["t"])
    return Session(name=name, landmarks_path=lm, frames=frames, segments=segments, meta=meta)


def auto_still_segments(
    session: Session,
    max_speed: float = 0.12,
    min_duration_ms: float = 1500.0,
    half_window_ms: float = 250.0,
) -> list[Segment]:
    """Find stretches where the upper body barely moves, for sessions without protocol segments.

    Positions of nose/shoulders/elbows/wrists (pose world landmarks) are averaged over
    ±`half_window_ms` first: MediaPipe's frame-to-frame noise alone moves a landmark at ~5 cm/s,
    which averaging removes while real motion persists. A frame is still while the fastest
    averaged landmark stays under `max_speed` (m/s). The default was set from one static test clip
    (p90 0.09 m/s) against vspark's recorded motion clips (medians 0.09-0.32 m/s), so this is a
    rough fallback; protocol segments from the recorder are the reliable source.
    """
    t = session.times
    pose = session.stream("pose")
    idx = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    X = np.array([p[idx, :3] if p is not None else np.full((len(idx), 3), np.nan) for p in pose])
    still = np.zeros(len(t), dtype=bool)
    if len(t) < 2:
        return []
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg = np.array([np.nanmean(X[(t >= tt - half_window_ms) & (t <= tt + half_window_ms)], axis=0) for tt in t])
        dt = np.diff(t) / 1000.0
        speed = np.nanmax(np.linalg.norm(np.diff(avg, axis=0), axis=2), axis=1) / np.where(dt > 0, dt, np.nan)
    still[1:] = speed < max_speed  # NaN (missing pose) compares False
    segs: list[Segment] = []
    k = 0
    while k < len(t):
        if not still[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(t) and still[j + 1]:
            j += 1
        if t[j] - t[k] >= min_duration_ms:
            segs.append(Segment("still", t[k], t[j]))
        k = j + 1
    return segs


@dataclass
class PoseTrack:
    """System output: per-frame VRM bone rotations. `bones[name]` is (N, 4) with NaN where absent."""

    times: np.ndarray  # ms
    bones: dict[str, np.ndarray]

    @classmethod
    def from_frames(cls, frames: list[dict]) -> "PoseTrack":
        times = np.array([f["t"] for f in frames], dtype=float)
        names = sorted({b for f in frames for b in f["bones"]})
        bones = {n: np.full((len(frames), 4), np.nan) for n in names}
        for i, f in enumerate(frames):
            for n, q in f["bones"].items():
                bones[n][i] = q
        return cls(times, bones)

    @classmethod
    def load(cls, path: str | Path) -> "PoseTrack":
        lines = Path(path).read_text().splitlines()
        return cls.from_frames([json.loads(line) for line in lines if line.strip()])
