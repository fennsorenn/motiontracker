"""Rendered synthetic sessions: exact ground truth for the whole system with the real detector.

    uv run python -m mttrain.render prepare --out data/m1/renders/v1 [--n 6] [--seconds 20] [--seed 0] [--rest]
                                           [--fovs 50,65,80,95]   # each sequence once per field of view
                                           [--contact]            # designed contact choreography
    uv run python -m mttrain.render video   --out data/m1/renders/v1      # Blender + ffmpeg
    uv run python -m mttrain.render replay  --out data/m1/renders/v1      # headless recorder -> landmarks
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
from mtcore.mhr import limits, load, skeleton_state
from mtcore.paths import AVATARS, REPO_ROOT
from mtcore.vrm import load_vrm

from . import choreo
from .camera import sample_desk_camera
from .dataset import FITS, clip_table
from .infer import OUTPUT_BONES, R0, mat_to_quat
from .seated import REST, seat
from .skeleton import quat_to_mat

FPS = 30.0
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


def prepare(out, n, seconds, seed, avatar="AvatarSample_D", rest_mode=False, fovs=None, contact_mode=False):
    """`rest_mode`: a held CMU posture with both arms in rest postures (lap, desk, hanging; random per arm)
    plus the seated idle motion (breathing, sway): nearly still ground truth, for noise at rest.
    `fovs`: horizontal fields of view (deg); each sequence is written once per value (seqNN_fovFF), same
    motion, appearance and camera yaw, height, roll and framing; the distance follows the field of view.
    `contact_mode`: designed contact choreography (mttrain.choreo) on held-out subjects' bodies; the
    contact intervals go to contacts.json. Also meant as follow-along reference videos for recordings."""
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
        elif rest_mode:
            held = np.repeat(mps[key][s0 + rng.integers(T)][None], T, 0)
            pair = tuple(rng.choice(list(REST), 2))
            mp = seat(rng, pn, held, FPS, lim, force_rest=pair)
        else:
            mp = seat(rng, pn, mps[key][s0:s0 + T], FPS, lim)
        states = skeleton_state(ch, mp)
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
            _write(Path(out) / name, verts, faces, cam, mp, skin, wall, key, states, conv, T)
            if contacts is not None:
                (Path(out) / name / "contacts.json").write_text(json.dumps(contacts, indent=1))
        tally = {v: round(sum(c["t1"] - c["t0"] for c in contacts if c["variant"] == v) / 1000, 1)
                 for v in ("palms-together", "hand-to-face", "near-face")} if contacts else None
        print(f"seq{i:02d}: clip {key} from frame {s0}, {T} frames" + (f", rest {pair}" if rest_mode else "")
              + (f", fov {fovs}" if fovs else "") + (f", contact s {tally}" if tally else ""), flush=True)


def _write(d, verts, faces, cam, mp, skin, wall, key, states, conv, T):
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "seq.npz", verts=verts.astype(np.float32), faces=faces, R=cam.R, t=cam.t,
                        fx=cam.fx, mp=mp.astype(np.float32), fps=FPS, skin=skin, wall=wall, clip=key)
    _write_gt(d, states, cam.R, conv, T)


def _write_gt(d, states, cam_R, conv, T):
    # ground truth, camera-relative like the model's output: G' = R0^T R_cam G
    G = quat_to_mat(states[..., 3:7])
    Gc = R0.T @ cam_R @ G
    st = states.copy()
    st[..., 3:7] = mat_to_quat(Gc)
    bones = conv.locals(st)
    with open(d / "gt.jsonl", "w") as f:
        for k in range(T):
            f.write(json.dumps({"t": round(k / FPS * 1000, 1),
                                "bones": {b: bones[b][k].round(6).tolist() for b in OUTPUT_BONES if b in bones}}) + "\n")


def gt(out, avatar="AvatarSample_D"):
    """Rewrite every sequence's gt.jsonl from its stored model parameters and camera (no re-render)."""
    ch = load()
    jn = list(ch.skeleton.joint_names)
    conv = MhrToVrm(jn, skeleton_state(ch, np.zeros(len(ch.parameter_transform.names))),
                    load_vrm(AVATARS / f"{avatar}.vrm")[2])
    for d in sorted(Path(out).glob("seq*")):
        z = np.load(d / "seq.npz")
        _write_gt(d, skeleton_state(ch, z["mp"].astype(np.float64)), z["R"], conv, len(z["mp"]))
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


def replay(out):
    port = _free_port()
    server = subprocess.Popen(["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1", "-d", str(REPO_ROOT)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for d in sorted(Path(out).glob("seq*")):
            if (d / "landmarks.jsonl").exists():
                continue
            rel = (d / "video.webm").resolve().relative_to(REPO_ROOT)
            url = f"http://127.0.0.1:{port}/tools/recorder/index.html?video=/{rel}&test"
            tmp = d / "replay"
            tmp.mkdir(exist_ok=True)
            seconds = len(np.load(d / "seq.npz")["mp"]) / FPS
            subprocess.run(["node", str(DRIVER), url, str(tmp), str(int(seconds * 2 + 30))], check=True)
            for f in ("session.json", "landmarks.jsonl"):
                shutil.move(tmp / f, d / f)
            if (d / "contacts.json").exists():  # true contact intervals as protocol segments
                meta = json.loads((d / "session.json").read_text())
                meta["segments"] = [g for g in meta.get("segments", []) if g["label"] != "contact"] \
                    + json.loads((d / "contacts.json").read_text())
                (d / "session.json").write_text(json.dumps(meta, indent=1))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{d.name}: landmarks ok", flush=True)
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
    ap.add_argument("--contact", action="store_true", help="designed contact choreography (mttrain.choreo)")
    ap.add_argument("--fovs", type=lambda v: [float(x) for x in v.split(",")], default=None,
                    help="comma-separated horizontal fields of view (deg): each sequence once per value")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.out, a.n, a.seconds, a.seed, rest_mode=a.rest, fovs=a.fovs, contact_mode=a.contact)
    elif a.step == "video":
        video(a.out, a.threads)
    elif a.step == "gt":
        gt(a.out)
    else:
        replay(a.out)


if __name__ == "__main__":
    main()
