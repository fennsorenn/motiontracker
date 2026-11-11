"""How well do head landmarks follow the head's rotation relative to the chest, and the pose's hand
points the wrist bend? MediaPipe on rendered sessions (mttrain.render) against synthetic observations of
the same motion and camera, both against the truth. Used to calibrate mttrain.mpstyle's head and hand
response (HEAD_GAIN, HEAD_NOISE_DEG, HAND_GAIN, HAND_NOISE_DEG).

    uv run python -m mttrain.headcal [--sets v1 fov1 rest1 contact1] [--seed 0]
                                     [--gain world image] [--noise world image]   # try other head values
                                     [--session sessions/<s>]   # also: pose vs hand model on a recording

Wrist bend: image = signed in-plane angle of wrist -> index/pinky midpoint minus elbow -> wrist; world =
3D angle between the two. On a recording there is no truth: the pose's bend is compared with the hand
model's (wrist -> middle knuckle, against the pose forearm), whose own error lowers the slope.

Measures, per axis (turn, nod, tilt), slope and correlation of observed against true:
- world: head-vs-chest rotation from 3D landmarks (chest frame from shoulders and hips, head frame from
  ears, eyes and mouth corners), observed world landmarks vs the true 3D landmarks;
- image: head-vs-shoulder cues in the image (turn and nod: nose offset from the ear midpoint along and
  across the ear line over the ear distance; tilt: ear line angle minus shoulder line angle), observed
  image landmarks vs the true landmarks projected through the same camera.
Slope 1 / correlation 1: the observation follows the head fully.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mtcore.landmarks import MP33, LandmarkModel
from mtcore.mhr import load, skeleton_state
from mtcore.paths import DATA

from .camera import Camera
from . import mpstyle as ms
from .mpstyle import Style
from .noise import NoiseConfig
from .synth import synthesize

I = MP33.index
AXES = ("turn", "nod", "tilt")
ASPECT = np.array([4 / 3, 1.0])  # recorder frames are 640 x 480


def world_angles(P):  # noqa: D103 (same frames as mpstyle.landmark_frames)
    """(T, 33, 3) -> head-vs-chest (turn, nod, tilt) deg (T, 3)."""
    def fr(xa, up):
        x = xa / np.linalg.norm(xa, axis=-1, keepdims=True)
        z = np.cross(x, up)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        return np.stack([x, np.cross(z, x), z], -1)
    C = fr(P[:, I("left_shoulder")] - P[:, I("right_shoulder")],
           P[:, [I("left_shoulder"), I("right_shoulder")]].mean(1) - P[:, [I("left_hip"), I("right_hip")]].mean(1))
    H = fr(P[:, I("left_ear")] - P[:, I("right_ear")],
           P[:, [I("left_eye"), I("right_eye")]].mean(1) - P[:, [I("mouth_left"), I("mouth_right")]].mean(1))
    rel = np.einsum("tji,tjk->tik", C, H)
    f, x = rel[:, :, 2], rel[:, :, 0]
    return np.degrees(np.stack([np.arctan2(f[:, 0], f[:, 2]), np.arcsin(np.clip(-f[:, 1], -1, 1)),
                                np.arctan2(x[:, 1], x[:, 0])], 1))


def image_cues(P):
    """(T, 33, 2+) image landmarks -> (turn, nod, tilt) cues (T, 3); turn/nod in ear distances, tilt deg."""
    P = P[..., :2] * ASPECT
    ear = P[:, I("left_ear")] - P[:, I("right_ear")]
    sh = P[:, I("left_shoulder")] - P[:, I("right_shoulder")]
    tilt = (np.degrees(np.arctan2(ear[:, 1], ear[:, 0]) - np.arctan2(sh[:, 1], sh[:, 0])) + 180) % 360 - 180
    L = np.linalg.norm(ear, axis=1)
    e = ear / L[:, None]
    n = np.stack([-e[:, 1], e[:, 0]], 1)
    d = P[:, I("nose")] - 0.5 * (P[:, I("left_ear")] + P[:, I("right_ear")])
    return np.stack([(d * e).sum(1) / L, (d * n).sum(1) / L, tilt], 1)


SIDES = {"left": (13, 15, 19, 17), "right": (14, 16, 20, 18)}  # elbow, wrist, index, pinky


def _wrap(x):
    return (x + 180) % 360 - 180


def image_bend(P, side):
    el, wr, ix, pk = SIDES[side]
    P = P[..., :2] * ASPECT
    a = lambda v: np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return _wrap(a(0.5 * (P[:, ix] + P[:, pk]) - P[:, wr]) - a(P[:, wr] - P[:, el]))


def world_bend(P, side):
    el, wr, ix, pk = SIDES[side]
    f, h = P[:, wr] - P[:, el], 0.5 * (P[:, ix] + P[:, pk]) - P[:, wr]
    c = (f * h).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(h, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _bends(Tw, Ti, Ow, Oi):
    """-> {'hand image': (true, obs), 'hand world': ...}, both sides stacked, as (N, 1) columns."""
    out = {}
    for space, f, T, O in (("hand image", image_bend, Ti, Oi), ("hand world", world_bend, Tw, Ow)):
        out[space] = (np.concatenate([f(T, s) for s in SIDES])[:, None], np.concatenate([f(O, s) for s in SIDES])[:, None])
    return out


def session_hands(session):
    """Recorded session: pose bend vs hand-model bend per side -> printed slope/corr."""
    rows = [json.loads(line) for line in open(Path(session) / "landmarks.jsonl") if line.strip()]
    for side, hk in (("left", "leftHand2d"), ("right", "rightHand2d")):
        el, wr = SIDES[side][:2]
        P = np.array([r["pose2d"] for r in rows if r.get("pose2d") and r.get(hk)])
        H = np.array([r[hk] for r in rows if r.get("pose2d") and r.get(hk)])[..., :2] * ASPECT
        if not len(P):
            continue
        a = lambda v: np.degrees(np.arctan2(v[..., 1], v[..., 0]))
        Pp = P[..., :2] * ASPECT
        hm = _wrap(a(H[:, 9] - H[:, 0]) - a(Pp[:, wr] - Pp[:, el]))
        f = fit(hm[:, None].repeat(3, 1), image_bend(P, side)[:, None].repeat(3, 1))[0]
        print(f"  {side:5s} n={len(P):3d} hand-model bend p5..p95 {np.percentile(hm, 5):+4.0f}..{np.percentile(hm, 95):+4.0f}"
              f"  pose vs hand model slope {f[0]:+.2f} corr {f[1]:.2f}")


def fit(true, obs):
    """Per column (axis): slope of obs on true and correlation, (C, 2)."""
    out = []
    for i in range(true.shape[1]):
        x, y = true[:, i] - true[:, i].mean(), obs[:, i] - obs[:, i].mean()
        out.append(((x * y).sum() / max((x * x).sum(), 1e-12), np.corrcoef(x, y)[0, 1]))
    return np.array(out)


def collect(seq_dir, ch, lm, rng, style=None):
    """One rendered sequence -> {source: {space: (true (T, 3), observed (T, 3))}}, sources 'mediapipe'
    and 'synthetic'."""
    z = np.load(seq_dir / "seq.npz")
    fps = float(z["fps"])
    cam = Camera(z["R"], z["t"], float(z["fx"]))
    mp = z["mp"].astype(np.float64)
    true_lm = lm(skeleton_state(ch, mp)) / 100.0
    tw, ti, ow, oi = [], [], [], []
    for line in open(seq_dir / "landmarks.jsonl"):
        r = json.loads(line)
        if not r.get("pose") or not r.get("pose2d"):
            continue
        k = int(round(r["t"] / 1000 * fps))
        if k < len(true_lm):
            tw.append(true_lm[k]); ti.append(cam.project(true_lm[k])[0])
            ow.append(np.asarray(r["pose"])[:, :3]); oi.append(np.asarray(r["pose2d"])[:, :3])
    tw, ti, ow, oi = map(np.array, (tw, ti, ow, oi))
    out = {"mediapipe": {"world": (world_angles(tw), world_angles(ow)), "image": (image_cues(ti), image_cues(oi)),
                         **_bends(tw, ti, ow, oi)}}
    s = synthesize(rng, ch, lm, mp, src_fps=fps, fps=30.0, style=style or Style.sample(rng),
                   noise=NoiseConfig.sample(rng), camera=cam)
    true_s = lm(s["states"]) / 100.0
    ti_s = cam.project(true_s)[0]
    out["synthetic"] = {"world": (world_angles(true_s), world_angles(s["world"])),
                        "image": (image_cues(ti_s), image_cues(s["image"])),
                        **_bends(true_s, ti_s, s["world"], s["image"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["v1", "fov1", "rest1", "contact1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, nargs=2, metavar=("WORLD", "IMAGE"), help="override mpstyle.HEAD_GAIN")
    ap.add_argument("--noise", type=float, nargs=2, metavar=("WORLD", "IMAGE"), help="override HEAD_NOISE_DEG")
    ap.add_argument("--session", help="recorded session: also compare the pose's wrist bend with the hand model's")
    a = ap.parse_args()
    if a.gain:
        ms.HEAD_GAIN.update(world=a.gain[0], image=a.gain[1])
    if a.noise:
        ms.HEAD_NOISE_DEG.update(world=a.noise[0], image=a.noise[1])
    print(f"synthetic head response: gain {ms.HEAD_GAIN}, noise deg {ms.HEAD_NOISE_DEG}; hands: gain {ms.HAND_GAIN}, "
          f"noise deg {ms.HAND_NOISE_DEG} (per-sequence spread)")
    rng = np.random.default_rng(a.seed)
    ch = load()
    lm = LandmarkModel(ch, skeleton_state(ch, np.zeros(len(ch.parameter_transform.names))))
    pooled = {}
    for name in a.sets:
        acc = {}
        for d in sorted((DATA / "m1" / "renders" / name).glob("seq*")):
            for src, spaces in collect(d, ch, lm, rng).items():
                for space, (t, o) in spaces.items():
                    acc.setdefault((src, space), ([], []))
                    acc[(src, space)][0].append(t); acc[(src, space)][1].append(o)
        print(f"== {name}")
        for (src, space), (t, o) in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            t, o = np.concatenate(t), np.concatenate(o)
            pooled.setdefault((src, space), ([], []))
            pooled[(src, space)][0].append(t); pooled[(src, space)][1].append(o)
            f = fit(t, o)
            print(f"  {space:10s} {src:9s} " + "  ".join(f"{ax} {s:+.2f}/{c:.2f}" for ax, (s, c) in zip(AXES if len(f) == 3 else ["bend"], f)))
    print("== pooled (slope/corr)")
    for (src, space), (t, o) in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        f = fit(np.concatenate(t), np.concatenate(o))
        print(f"  {space:10s} {src:9s} " + "  ".join(f"{ax} {s:+.2f}/{c:.2f}" for ax, (s, c) in zip(AXES if len(f) == 3 else ["bend"], f)))
    if a.session:
        print(f"== {a.session}: wrist bend, pose vs hand model (image)")
        session_hands(a.session)


if __name__ == "__main__":
    main()
