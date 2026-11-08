"""Tracking systems the harness can run over a session's landmarks.

Built in:
  baseline           vspark's solver (backend output, one pose per detector frame)
  baseline-rendered  + vspark's frontend One Euro filter at a simulated 60 Hz render rate

Any other system can be evaluated from a precomputed output file (`name=path.jsonl`, one
`{t, bones: {vrmBone: [x, y, z, w]}}` per line). A name ending in `-rendered`
(`m1-rendered=path.jsonl`) runs the file through the same frontend render stage as
`baseline-rendered` first (the baseline CLI's filter-only `--poses` mode), so it can be compared
with `baseline-rendered`. A name ending in `-glide` (`m1-glide=path.jsonl`) is upsampled to 60 Hz
without smoothing instead (`upsample.glide`), for systems that replace the filter too.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .session import PoseTrack, Session
from .upsample import glide

REPO = Path(os.environ.get("MT_REPO", Path(__file__).resolve().parents[3]))
BUILTIN = ("baseline", "baseline-rendered")


@dataclass
class SystemResult:
    name: str
    track: PoseTrack
    notes: list[str]


def head_calibration_time(session: Session) -> float | None:
    """vspark users capture the head neutral while sitting still; use the middle of the first
    'still' protocol segment, if the session has one."""
    still = session.segments_labeled("still")
    return (still[0].t0 + still[0].t1) / 2 if still else None


def run_baseline(session: Session, stage: str = "solver", calibrate: bool = True) -> SystemResult:
    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "poses.jsonl"
        cmd = [
            "pnpm", "--dir", str(REPO / "baseline"), "-s", "run-baseline",
            "--input", str(session.landmarks_path.resolve()),
            "--output", str(out),
            "--stage", stage,
        ]
        cal = head_calibration_time(session) if calibrate else None
        if cal is not None:
            cmd += ["--capture-head-at", str(cal)]
            notes.append(f"head calibration captured at t={cal:.0f} ms")
        else:
            notes.append("no head calibration")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return SystemResult("baseline" if stage == "solver" else "baseline-rendered", PoseTrack.load(out), notes)


def run_rendered(name: str, path: str | Path) -> SystemResult:
    """A precomputed pose track through vspark's frontend One Euro filter at 60 Hz."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "poses.jsonl"
        cmd = [
            "pnpm", "--dir", str(REPO / "baseline"), "-s", "run-baseline",
            "--poses", str(Path(path).resolve()),
            "--output", str(out),
            "--stage", "rendered",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return SystemResult(name, PoseTrack.load(out), [f"precomputed from {path}, through the render stage"])


def run_system(spec: str, session: Session, calibrate: bool = True) -> SystemResult:
    if "=" in spec:
        name, path = spec.split("=", 1)
        if name.endswith("-rendered"):
            return run_rendered(name, path)
        if name.endswith("-glide"):
            return SystemResult(name, glide(PoseTrack.load(path)), [f"precomputed from {path}, glide-upsampled to 60 Hz"])
        return SystemResult(name, PoseTrack.load(path), [f"precomputed from {path}"])
    if spec == "baseline":
        return run_baseline(session, "solver", calibrate)
    if spec == "baseline-rendered":
        return run_baseline(session, "rendered", calibrate)
    raise ValueError(f"unknown system {spec!r}; built in: {', '.join(BUILTIN)}, or name=path.jsonl (name-rendered=… for the render filter, name-glide=… for 60 Hz upsampling)")
