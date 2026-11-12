"""Rendered synthetic sessions: exact ground truth for the whole system with the real detector.

    uv run python -m mttrain.render prepare --out data/m1/renders/v1 [--n 6] [--seconds 20] [--seed 0] [--rest]
                                           [--head]               # designed head rotations, torso still
                                           [--fovs 50,65,80,95]   # each sequence once per field of view
                                           [--contact]            # designed contact choreography
                                           [--hands]              # procedural finger motion, hands in view
    uv run python -m mttrain.render video   --out data/m1/renders/v1      # Blender + ffmpeg
    uv run python -m mttrain.render replay  --out data/m1/renders/v1      # headless recorder -> landmarks
                                           [--mode image]         # IMAGE running mode -> landmarks-image.jsonl
                                                                  # (still smoothed in 0.10.17: see recorder.js)
    uv run python -m mttrain.render gt      --out data/m1/renders/v1      # rewrite gt.jsonl from seq.npz
                                                                          # (after a conversion change)

Held-out CMU motion (validation subjects) goes through the seated pipeline at 30 fps; the skinned MHR
mesh is rendered in Blender (tools/render/render_blender.py) from a desk camera sampled like the
training cameras; the video replays through the recorder in headless Chrome, so the landmarks come
from the exact MediaPipe build used live. The recorder mirrors its input before detection, so the
video is mirrored first: MediaPipe then sees the unmirrored render, and its labels match the
training convention (an unmirrored camera on the MHR performer; the avatar mirrors the user).

Per sequence (data/m1/renders/<set>/seqNN/, CMU-derived: never commit):
  seq.npz        vertices (T, V, 3) MHR metres, faces, camera R/t/fx, model parameters, fps
  gt.jsonl       ground-truth VRM pose track ({t, bones}), camera-relative like the model's output
  fingers.npz    (--hands) finger parameters per frame (MHR, `mttrain.hands.skeleton.PARAMS` order) of
                 both hands and the key-pose labels; gt.jsonl then also has the VRM finger bones
  contacts.json  (--contact) contact intervals as session segments ('contact' palms-together /
                 hand-to-face with settleMs 0 and the true 7b distance trueCm; 'near-contact' near-face);
                 replay merges them into session.json for criterion 7b
  frames/, video.webm, session.json, landmarks.jsonl
Evaluate with `mteval evaluate <seq dir> --gt <seq dir>/gt.jsonl --system ...`.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import numpy as np

from mtcore.convert import MhrToVrm
from mtcore.landmarks import MP33, LandmarkModel
from mtcore.lifter_pose import R0, mat_to_quat
from mtcore.mhr import limits, load, skeleton_state
from mtcore.paths import AVATARS, REPO_ROOT
from mtcore.vrm import load_vrm

from . import choreo
from .camera import sample_desk_camera
from .dataset import FITS, clip_table
from .infer import OUTPUT_BONES
from .seated import REST, seat
from .skeleton import quat_to_mat

FPS = 30.0
FINGER_WORDS = ("Thumb", "Index", "Middle", "Ring", "Little")
BLENDER = os.environ.get("MT_BLENDER", "blender")
SCRIPT = REPO_ROOT / "tools" / "render" / "render_blender.py"
DRIVER = REPO_ROOT / "tools" / "recorder" / "test" / "drive_headless.mjs"


def _contact_distances(ch, lm, mp, seg):
    """True wrist-wrist / nearer wrist-nose distance (cm, median) during a segment: criterion 7b's
    quantities on the true body (they are not zero at contact: the wrists sit behind the palms)."""
    k0, k1 = int(seg["t0"] / 1000 * FPS), int(seg["t1"] / 1000 * FPS)
    X = lm(skeleton_state(ch, mp[k0:k1]))
    lw, rw, nose = X[:, MP33.index("left_wrist")], X[:, MP33.index("right_wrist")], X[:, MP33.index("nose")]
    if seg["variant"] == "palms-together":
        return float(np.median(np.linalg.norm(lw - rw, axis=1)))
    return float(np.median(np.minimum(np.linalg.norm(lw - nose, axis=1), np.linalg.norm(rw - nose, axis=1))))


# head choreography (head_mode): (turn, nod, tilt) targets in degrees of the head relative to the chest, held
# for a moment and reached with minimum-jerk moves; split between the neck and head joints. Positive turn =
# MHR twist +, nod = bend + (down), tilt = lean +. Reaches 60 deg turns, 35 deg nods, 30 deg tilts and
# combinations: the range MediaPipe compresses (mttrain.headcal).
HEAD_POSES = [(0, 0, 0), (45, 0, 0), (-45, 0, 0), (0, 0, 0), (0, 30, 0), (0, -25, 0), (0, 0, 0), (0, 0, 25),
              (0, 0, -25), (0, 0, 0), (60, 0, 0), (-60, 0, 0), (0, 0, 0), (35, 25, 0), (-35, 25, 0), (-35, -20, 0),
              (35, -20, 0), (0, 0, 0), (30, 0, 25), (-30, 0, -25), (20, 15, -20), (0, 0, 0)]


def head_choreography(rng, pn, mp, fps, limits):
    """Overwrite the neck/head parameters of a held seated posture with the HEAD_POSES sequence."""
    P = {n: i for i, n in enumerate(pn)}
    n = len(mp)
    t = np.arange(n) / fps
    step = t[-1] / (len(HEAD_POSES) - 1)  # seconds per target: move (60 %) then hold
    keys = np.radians(np.array(HEAD_POSES, dtype=float)) / np.radians(17.2 / 0.3)  # deg -> rad of joint parameter
    out = mp.copy()
    seg = np.minimum((t / step).astype(int), len(HEAD_POSES) - 2)
    u = np.clip((t - seg * step) / (0.6 * step), 0, 1)
    w = u ** 3 * (10 - 15 * u + 6 * u * u)
    cur = keys[seg] + (keys[seg + 1] - keys[seg]) * w[:, None]  # (n, 3) total turn, nod, tilt in parameter units
    from .seated import HEAD_SHARES, clamp
    for k, (neck, head, share) in enumerate(HEAD_SHARES):
        out[:, P[neck]] = share * cur[:, k]
        out[:, P[head]] = (1 - share) * cur[:, k]
    return clamp(pn, out, limits)


def prepare(out, n, seconds, seed, avatar="AvatarSample_D", rest_mode=False, fovs=None, contact_mode=False,
            head_mode=False, hands_mode=False):
    """`rest_mode`: a held CMU posture with both arms in rest postures (lap, desk, hanging; random per arm)
    plus the seated idle motion (breathing, sway): nearly still ground truth, for noise at rest.
    `head_mode`: as rest_mode, but the head runs the HEAD_POSES choreography relative to the still torso
    (no idle head motion): MediaPipe's head response against exact truth.
    `fovs`: horizontal fields of view (deg); each sequence is written once per value (seqNN_fovFF), same
    motion, appearance and camera yaw, height, roll and framing; the distance follows the field of view.
    `contact_mode`: designed contact choreography (mttrain.choreo) on held-out subjects' bodies; the
    contact intervals go to contacts.json. Also meant as follow-along reference videos for recordings.
    `hands_mode`: hands in use (mttrain.hands.stage: arms authored with IK, hands in front of the chest
    and face, palms any way) with procedural finger motion (mttrain.hands.motion), on held-out bodies;
    camera field of view 60-95 deg (the owner's cameras: about 70 and 90), drawn again until at least one
    hand is in the frame for 70% of the frames. Ground truth gains the finger bones."""
    rng = np.random.default_rng(seed)
    ch = load()
    pn = list(ch.parameter_transform.names)
    jn = list(ch.skeleton.joint_names)
    rest = skeleton_state(ch, np.zeros(len(pn)))
    lm = LandmarkModel(ch, rest)
    lim = limits()
    conv = MhrToVrm(jn, rest, load_vrm(AVATARS / f"{avatar}.vrm")[2])
    T = int(seconds * FPS)
    rows = [(k, w) for k, w in clip_table("val")]
    mps = {k: np.load(FITS / f"{k}.npz")["mp"] for k, _ in rows}
    keys = [k for k, _ in rows if len(mps[k]) >= T]
    w = np.array([w for k, w in rows if k in keys])
    faces = np.asarray(ch.mesh.faces, dtype=np.int32)
    # distinct clips (weighted like training), repeating only if there are fewer clips than sequences
    order = rng.choice(len(keys), size=min(n, len(keys)), replace=False, p=w / w.sum())
    print(f"{len(keys)} held-out clips are at least {seconds:.0f} s long", flush=True)
    for i in range(n):
        key = keys[order[i % len(order)]]
        s0 = rng.integers(0, len(mps[key]) - T + 1)
        contacts = None
        if contact_mode:
            mp, contacts, drift = choreo.generate(rng, ch, lm, pn, mps[key][s0].astype(np.float64), T, FPS, lim)
            for c, dr in zip(contacts, drift):
                if c["label"] == "contact":
                    c["trueCm"] = round(_contact_distances(ch, lm, mp, c), 1)
                print(f"  {c['label']} {c['variant']} {c['t0'] / 1000:.1f}-{c['t1'] / 1000:.1f} s, hold drift {dr:.2f} cm"
                      + (f", true distance {c['trueCm']} cm" if "trueCm" in c else ""), flush=True)
        elif rest_mode or head_mode:
            held = np.repeat(mps[key][s0 + rng.integers(T)][None], T, 0)
            pair = tuple(rng.choice(list(REST), 2))
            mp = seat(rng, pn, held, FPS, lim, force_rest=pair, moves=False)
            if head_mode:
                mp = head_choreography(rng, pn, mp, FPS, lim)
        elif hands_mode:
            mp, cam, cam_state, key, s0, fingers = _hands_take(rng, ch, lm, pn, lim, mps, keys, w, T)
        else:
            mp = seat(rng, pn, mps[key][s0:s0 + T], FPS, lim)
        states = skeleton_state(ch, mp)
        if not hands_mode:
            cam_state = rng.bit_generator.state
            cam = sample_desk_camera(rng, lm(states) / 100.0)
        verts = np.asarray(ch.skin_points(states.astype(np.float32))) / 100.0
        skin = np.clip(np.array([0.80, 0.60, 0.48]) * rng.uniform(0.55, 1.1) + rng.normal(0, 0.03, 3), 0.05, 1.0)
        wall = rng.uniform(0.45, 0.9, 3)
        for fov in (fovs or [None]):
            if fov is not None:
                crng = np.random.default_rng()
                crng.bit_generator.state = cam_state
                cam = sample_desk_camera(crng, lm(states) / 100.0, fov=fov)
            name = f"seq{i:02d}" + (f"_fov{fov:.0f}" if fov is not None else "")
            _write(Path(out) / name, verts, faces, cam, mp, skin, wall, key, states, conv, T, fingers=hands_mode)
            if hands_mode:
                np.savez_compressed(Path(out) / name / "fingers.npz", **fingers)
            if contacts is not None:
                (Path(out) / name / "contacts.json").write_text(json.dumps(contacts, indent=1))
        tally = {v: round(sum(c["t1"] - c["t0"] for c in contacts if c["variant"] == v) / 1000, 1)
                 for v in ("palms-together", "hand-to-face", "near-face")} if contacts else None
        print(f"seq{i:02d}: clip {key} from frame {s0}, {T} frames" + (f", rest {pair}" if rest_mode or head_mode else "")
              + (f", fov {fovs}" if fovs else "") + (f", contact s {tally}" if tally else ""), flush=True)


def _hands_take(rng, ch, lm, pn, lim, mps, keys, w, T, tries=40):
    """Hands-in-use arms with fingers moving and a camera that sees at least one hand 70% of the time."""
    from .hands import stage
    from .hands.motion import HandMotion
    from .hands.skeleton import HandFK
    global _HAND_MOTION
    if "_HAND_MOTION" not in globals():
        _HAND_MOTION = HandMotion(HandFK(ch))
    hm = _HAND_MOTION
    wrists = [MP33.index("left_wrist"), MP33.index("right_wrist")]
    tips = [MP33.index("left_index"), MP33.index("right_index")]
    best = None
    for _ in range(tries):
        key = keys[rng.choice(len(keys), p=w / w.sum())]
        if len(mps[key]) < T:
            continue
        s0 = rng.integers(0, len(mps[key]) - T + 1)
        mp = stage.generate(rng, ch, lm, pn, mps[key][s0].astype(np.float64), T, FPS, lim)
        fingers = {}
        for side in "lr":
            q, labels = hm.sequence(rng, T / FPS, FPS)
            mp[:, hm.fk.idx[side]] = q
            fingers[side] = q.astype(np.float32)
            fingers[f"{side}_labels"] = np.array(json.dumps(labels))
        X = lm(skeleton_state(ch, mp)) / 100.0
        cam_state = rng.bit_generator.state
        cam = sample_desk_camera(rng, X, fov=rng.uniform(60, 95))
        uv, front = cam.project(X)
        inside = (uv[..., 0] > 0.03) & (uv[..., 0] < 0.97) & (uv[..., 1] > 0.03) & (uv[..., 1] < 0.97) & front
        seen = (inside[:, wrists] & inside[:, tips]).any(1).mean()
        if best is None or seen > best[0]:
            best = (seen, mp, cam, cam_state, key, s0, fingers)
        if seen >= 0.7:
            break
    seen, *rest = best
    print(f"  hands in view {100 * seen:.0f}% of frames", flush=True)
    return tuple(rest)


def _write(d, verts, faces, cam, mp, skin, wall, key, states, conv, T, fingers=False):
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "seq.npz", verts=verts.astype(np.float32), faces=faces, R=cam.R, t=cam.t,
                        fx=cam.fx, mp=mp.astype(np.float32), fps=FPS, skin=skin, wall=wall, clip=key, fingers=fingers)
    _write_gt(d, states, cam.R, conv, T, fingers)


def _write_gt(d, states, cam_R, conv, T, fingers=False):
    # ground truth, camera-relative like the model's output: G' = R0^T R_cam G
    G = quat_to_mat(states[..., 3:7])
    Gc = R0.T @ cam_R @ G
    st = states.copy()
    st[..., 3:7] = mat_to_quat(Gc)
    bones = conv.locals(st)
    names = OUTPUT_BONES + ([b for b in bones if any(f in b for f in FINGER_WORDS)] if fingers else [])
    with open(d / "gt.jsonl", "w") as f:
        for k in range(T):
            f.write(json.dumps({"t": round(k / FPS * 1000, 1),
                                "bones": {b: bones[b][k].round(6).tolist() for b in names if b in bones}}) + "\n")


def gt(out, avatar="AvatarSample_D"):
    """Rewrite every sequence's gt.jsonl from its stored model parameters and camera (no re-render)."""
    ch = load()
    jn = list(ch.skeleton.joint_names)
    conv = MhrToVrm(jn, skeleton_state(ch, np.zeros(len(ch.parameter_transform.names))),
                    load_vrm(AVATARS / f"{avatar}.vrm")[2])
    for d in sorted(Path(out).glob("seq*")):
        z = np.load(d / "seq.npz")
        _write_gt(d, skeleton_state(ch, z["mp"].astype(np.float64)), z["R"], conv, len(z["mp"]),
                  bool(z["fingers"]) if "fingers" in z else False)
        print(f"{d.name}: gt.jsonl rewritten", flush=True)


def video(out, threads=8):
    for d in sorted(Path(out).glob("seq*")):
        frames = d / "frames"
        if not (d / "video.webm").exists():
            if not frames.exists() or len(list(frames.glob("*.png"))) < len(np.load(d / "seq.npz")["mp"]):
                subprocess.run([BLENDER, "-b", "-t", str(threads), "--python", str(SCRIPT), "--",
                                str(d / "seq.npz"), str(frames)], check=True, stdout=subprocess.DEVNULL)
            # mirrored, so the recorder's own mirroring hands MediaPipe the unmirrored render
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                            "-vf", "hflip,format=yuv420p", "-c:v", "libvpx-vp9", "-b:v", "4M",
                            "-threads", str(threads), str(d / "video.webm")], check=True)
        print(f"{d.name}: video ok", flush=True)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def replay(out, mode="video"):
    """Recorder replay of every sequence's video -> landmarks.jsonl + session.json. mode='image': the
    detector in IMAGE running mode, frame-stepped -> landmarks-image.jsonl and session-image.json beside
    the default ones (caution: still landmark-smoothed in tasks-vision 0.10.17, see tools/recorder)."""
    sfx = "" if mode == "video" else f"-{mode}"
    port = _free_port()
    server = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "-d", str(REPO_ROOT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for d in sorted(Path(out).glob("seq*")):
            if (d / f"landmarks{sfx}.jsonl").exists():
                continue
            rel = (d / "video.webm").resolve().relative_to(REPO_ROOT)
            url = f"http://127.0.0.1:{port}/tools/recorder/index.html?video=/{rel}&test" + (f"&mode={mode}" if sfx else "")
            tmp = d / "replay"
            tmp.mkdir(exist_ok=True)
            seconds = len(np.load(d / "seq.npz")["mp"]) / FPS
            # stepped image-mode replays run at detection speed, not real time
            subprocess.run(["node", str(DRIVER), url, str(tmp), str(int(seconds * (2 if not sfx else 40) + 30))], check=True)
            for f in ("session", "landmarks"):
                ext = ".json" if f == "session" else ".jsonl"
                shutil.move(tmp / f"{f}{ext}", d / f"{f}{sfx}{ext}")
            if (d / "contacts.json").exists():  # true contact intervals as protocol segments
                meta = json.loads((d / f"session{sfx}.json").read_text())
                meta["segments"] = [g for g in meta.get("segments", []) if g["label"] != "contact"] \
                    + json.loads((d / "contacts.json").read_text())
                (d / f"session{sfx}.json").write_text(json.dumps(meta, indent=1))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{d.name}: landmarks{sfx} ok", flush=True)
    finally:
        server.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "video", "replay", "gt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rest", action="store_true", help="held posture, arms resting (noise at rest)")
    ap.add_argument("--head", action="store_true", help="held posture, arms resting, designed head rotations")
    ap.add_argument("--contact", action="store_true", help="designed contact choreography (mttrain.choreo)")
    ap.add_argument("--hands", action="store_true", help="procedural finger motion, hands kept in view")
    ap.add_argument("--mode", choices=("video", "image"), default="video",
                    help="replay: detector running mode (image: frame-stepped; still smoothed in 0.10.17)")
    ap.add_argument("--fovs", type=lambda v: [float(x) for x in v.split(",")], default=None,
                    help="comma-separated horizontal fields of view (deg): each sequence once per value")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.out, a.n, a.seconds, a.seed, rest_mode=a.rest, fovs=a.fovs, contact_mode=a.contact, head_mode=a.head,
                hands_mode=a.hands)
    elif a.step == "video":
        video(a.out, a.threads)
    elif a.step == "gt":
        gt(a.out)
    else:
        replay(a.out, a.mode)


if __name__ == "__main__":
    main()
