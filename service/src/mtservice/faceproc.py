"""Offline face post-processing, exactly as the runtime does it live (`mtcore.face_filter`): a landmarks dump
(`--dump-landmarks`, raw blendshapes per detector frame) -> processed blendshapes at the output rate, for the avatar
videos (`tools/avatar-video/prepare.py --blendshapes NAME=OUT`).

    uv run --project service python -m mtservice.faceproc IN.jsonl OUT.jsonl [--hz 60] [--no-eye-filter] [--no-glide]

Prints before/after statistics of the eye shapes (resting level, left-right agreement, blink peaks).
"""

import argparse
import json
from pathlib import Path

from mtcore.face_filter import EyeFilter, Glide


def process(rows, eye_filter=True, hz=60.0, glide=True):
    """Detector frames [{t (ms), blendshapes}] -> output frames [{t (ms), blendshapes}] and the per-frame processed
    values (before the glide)."""
    eyes, gl = EyeFilter(), Glide(hz)
    frames, out = [], []
    for r in rows:
        bs = r.get("blendshapes")
        t = r["t"] / 1000.0
        if bs and eye_filter:
            bs = eyes.step(t, bs)
        frames.append({"t": r["t"], "blendshapes": bs})
        if not glide:
            out.append({"t": r["t"], "blendshapes": bs})
    if glide and frames:
        i, t, t_end = 0, frames[0]["t"] / 1000.0, frames[-1]["t"] / 1000.0
        while t <= t_end + 1e-9:
            while i < len(frames) and frames[i]["t"] / 1000.0 <= t:
                gl.push(frames[i]["t"] / 1000.0, frames[i]["blendshapes"])
                i += 1
            v = gl.at(t)
            out.append({"t": round(t * 1000, 1), "blendshapes": {k: round(x, 4) for k, x in v.items()} or None})
            t += 1.0 / hz
    return out, frames


def eye_stats(frames):
    import statistics as st
    rows = [f["blendshapes"] for f in frames if f.get("blendshapes")]
    res = {}
    for name in ("eyeBlink", "eyeSquint"):
        l = [r[f"{name}Left"] for r in rows]
        r_ = [r[f"{name}Right"] for r in rows]
        diff = [abs(a - b) for a, b in zip(l, r_)]
        res[name] = {"rest_p50": round(st.median(l + r_), 3), "p99": round(sorted(l + r_)[int(0.99 * (2 * len(l) - 1))], 3),
                     "left_right_diff_p50": round(st.median(diff), 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--no-eye-filter", action="store_true")
    ap.add_argument("--no-glide", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.inp) if line.strip()]
    out, frames = process(rows, not a.no_eye_filter, a.hz, not a.no_glide)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("raw      ", json.dumps(eye_stats(rows)))
    print("processed", json.dumps(eye_stats(frames)))
    print(f"{len(rows)} detector frames -> {len(out)} output frames -> {a.out}")


if __name__ == "__main__":
    main()
