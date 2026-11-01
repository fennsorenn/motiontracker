"""Compare recorded sessions with synthetic observations: proportions, jitter, visibility.

    uv run python -m mttrain.calibrate [SESSION_DIR ...] [--clips 40]

Real numbers come from `sessions/*` (default: all). Synthetic numbers run fitted CMU clips
(data/m1/fits) through the camera, bias and noise models. Use it to recalibrate `mpstyle.py` and
`noise.py` when new sessions are recorded.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.landmarks import MP33, LandmarkModel
from mtcore.mhr import load, skeleton_state
from mtcore.paths import DATA, REPO_ROOT

from .noise import outside_distance
from .mpstyle import Style
from .noise import NoiseConfig
from .synth import synthesize

I = MP33.index
RATIOS = {"eye to eye": ("left_eye", "right_eye"), "ear to ear": ("left_ear", "right_ear"),
          "mouth width": ("mouth_left", "mouth_right"), "upper arm": ("left_shoulder", "left_elbow"),
          "forearm": ("left_elbow", "left_wrist"), "hip width": ("left_hip", "right_hip"),
          "torso": ("left_shoulder", "left_hip"), "nose to shoulder": ("nose", "left_shoulder")}
JITTER = {"face": ["nose", "left_eye", "left_ear"], "shoulder": ["left_shoulder", "right_shoulder"],
          "elbow": ["left_elbow", "right_elbow"], "wrist": ["left_wrist", "right_wrist"]}
VIS_BINS = [-1, -0.2, -0.05, 0, 0.05, 0.15, 0.3, 3]


def ratios(W, im, vis):
    ok = (vis[:, I("left_elbow")] > 0.9) & (vis[:, I("left_wrist")] > 0.5)
    sw_w = np.linalg.norm(W[:, I("left_shoulder")] - W[:, I("right_shoulder")], axis=-1)
    imsq = im[..., :2] * [4 / 3, 1]
    sw_i = np.linalg.norm(imsq[:, I("left_shoulder")] - imsq[:, I("right_shoulder")], axis=-1)
    out = {}
    for n, (a, b) in RATIOS.items():
        rw = np.linalg.norm(W[:, I(a)] - W[:, I(b)], axis=-1) / sw_w
        ri = np.linalg.norm(imsq[:, I(a)] - imsq[:, I(b)], axis=-1) / sw_i
        out[n] = (float(np.median(rw[ok])) if ok.any() else np.nan, float(np.median(ri[ok])) if ok.any() else np.nan)
    return out


def jitter(W, im, still_runs):
    """Lateral/depth world jitter (mm) and image jitter (px at 320) on still runs: deviation from a 5-frame mean."""
    def hp(x):
        k = np.ones(5) / 5
        return x[2:-2] - np.stack([np.convolve(x[:, j], k, "valid") for j in range(x.shape[1])], 1)
    out = {}
    for g, names in JITTER.items():
        jw, ji = [], []
        for r in still_runs:
            if len(r) < 8:
                continue
            for n in names:
                jw.append(hp(W[r, I(n)]))
                ji.append(hp(im[r, I(n), :2]))
        if jw:
            jw, ji = np.concatenate(jw), np.concatenate(ji)
            out[g] = (float(jw[:, :2].std() * 1000), float(jw[:, 2].std() * 1000), float(ji.std() * 320))
    return out


def vis_curve(im, vis):
    d = outside_distance(im).ravel()
    v = vis.ravel()
    return [float(np.median(v[(d >= lo) & (d < hi)])) if ((d >= lo) & (d < hi)).any() else np.nan
            for lo, hi in zip(VIS_BINS[:-1], VIS_BINS[1:])]


def real_stats(paths):
    Ws, Is, Vs, runs, off = [], [], [], [], 0
    for p in paths:
        seg = json.loads((p / "session.json").read_text())["segments"]
        L = [json.loads(line) for line in (p / "landmarks.jsonl").read_text().splitlines() if line.strip()]
        L = [f for f in L if f.get("pose") and f.get("pose2d")]
        t = np.array([f["t"] for f in L])
        P = np.array([f["pose"] for f in L])
        Ws.append(P[..., :3])
        Vs.append(P[..., 3])
        Is.append(np.array([f["pose2d"] for f in L])[..., :3])
        for s in seg:
            if s["label"] == "still":
                runs.append(off + np.flatnonzero((t >= s["t0"] + 1000) & (t < s["t1"])))
        off += len(L)
    W, im, vis = np.concatenate(Ws), np.concatenate(Is), np.concatenate(Vs)
    return ratios(W, im, vis), jitter(W, im, runs), vis_curve(im, vis)


def synthetic_stats(n_clips, seed=0, raw=False):
    rng = np.random.default_rng(seed)
    ch = load()
    pn = list(ch.parameter_transform.names)
    lm = LandmarkModel(ch, skeleton_state(ch, np.zeros(len(pn))))
    files = sorted((DATA / "m1" / "fits").glob("*.npz"))
    files = [files[i] for i in rng.choice(len(files), min(n_clips, len(files)), replace=False)]
    quiet = NoiseConfig(jitter_scale=0.0, guess_sigma=0.0, vis_logit_sigma=0.0, glitch_rate=0.0, dropout_rate=0.0)
    Ws, Is, Vs = [], [], []
    for f in files:
        mp = np.load(f)["mp"]
        if raw:  # geometry only: no MediaPipe bias, no noise
            s = synthesize(rng, ch, lm, mp, fps=10.0, style=Style.identity(), noise=quiet)
        else:
            s = synthesize(rng, ch, lm, mp, fps=10.0)
        Ws.append(s["world"]); Is.append(s["image"]); Vs.append(s["vis"])
    # jitter: frozen poses (10 s each), so only the noise model moves the points
    Js, Ji = [], []
    for f in files:
        mp = np.load(f)["mp"]
        frozen = np.repeat(mp[rng.integers(len(mp))][None], 300, 0)
        s = synthesize(rng, ch, lm, frozen, fps=10.0, noise=quiet if raw else None,
                       style=Style.identity() if raw else None)
        Js.append(s["world"]); Ji.append(s["image"])
    jw, ji = np.concatenate(Js), np.concatenate(Ji)
    bounds = np.cumsum([0] + [len(x) for x in Js])
    runs = [np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
    W, im, vis = np.concatenate(Ws), np.concatenate(Is), np.concatenate(Vs)
    return ratios(W, im, vis), jitter(jw, ji, runs), vis_curve(im, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--clips", type=int, default=40)
    a = ap.parse_args()
    paths = a.sessions or sorted(p for p in (REPO_ROOT / "sessions").iterdir() if (p / "landmarks.jsonl").exists())
    print(f"real: {len(paths)} session(s); synthetic: {a.clips} CMU clips")
    (rr, rj, rv), (sr, sj, sv) = real_stats(paths), synthetic_stats(a.clips)
    raw = synthetic_stats(a.clips, raw=True)[0]
    print("\nratio to shoulder width; raw = synthetic geometry without bias or noise; factor = real / raw")
    print(f"{'':20s}{'real':>7s}{'synth':>7s}{'raw':>7s}{'factor':>8s}   {'real':>7s}{'synth':>7s}{'raw':>7s}{'factor':>8s}")
    print(f"{'':20s}{'--- world landmarks ---':>29s}   {'--- image landmarks ---':>29s}")
    for n in RATIOS:
        print(f"{n:20s}{rr[n][0]:7.3f}{sr[n][0]:7.3f}{raw[n][0]:7.3f}{rr[n][0] / raw[n][0]:8.2f}   "
              f"{rr[n][1]:7.3f}{sr[n][1]:7.3f}{raw[n][1]:7.3f}{rr[n][1] / raw[n][1]:8.2f}")
    print(f"\n{'still jitter':14s}{'real lat/depth mm, px':>26s}{'synth':>24s}")
    for g in JITTER:
        f = lambda d: "%5.1f %5.1f %5.2f" % d.get(g, (np.nan,) * 3)
        print(f"{g:14s}{f(rj):>26s}{f(sj):>24s}")
    print("\nvisibility by distance outside the frame")
    for (lo, hi), r, s in zip(zip(VIS_BINS[:-1], VIS_BINS[1:]), rv, sv):
        print(f"  {lo:5.2f}..{hi:4.2f}: real {r:.2f}  synth {s:.2f}")


if __name__ == "__main__":
    main()
