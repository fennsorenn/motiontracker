"""mtservice: camera or video -> MediaPipe Holistic -> learned models -> VMC.

    mtservice [--camera N | --video FILE [--realtime]] [--vmc HOST:PORT] [--merger DIR | --no-merger [--bundle DIR]]
              [--task FILE] [--avatar-vrm FILE] [--no-retarget] [--face-mesh full|lite] [--no-face] [--face-rate HZ] [--no-eye-filter]
              [--no-fingers] [--no-legs] [--stdio] [--dump-poses FILE]
              [--dump-landmarks FILE] [--no-head3d] [--list-cameras]

Models: by default the merger bundle (`mttrain.merger.export`, `mtcore.merged`): body model, pair hand model, face-mesh
head pose and the merger -> body, leg and finger bones, hip position and up direction. --no-merger: the M1 lifter
alone (--bundle, upper body) with hands.py's fingers, as before 2026-11-14 (also used when the default merger
bundle is missing).
--no-legs: no leg bones (receivers without a lower-body mask). Hip position and up direction go out as /MT/Root and
/MT/Up (vmc.py).
--avatar-vrm: the bones are computed for this avatar's rest pose and, with the merger bundle, retargeted to its
proportions (`mtcore.relretarget`: hands keep their relations to the body and each other; the human side is MHR's
mean body, the bundle's `retarget.npz`). Without a VRM: the default avatar's rest pose and plain rotation copy (as
with an avatar given as rest positions only). --no-retarget: rotation copy even with a VRM.

--face-mesh: MediaPipe face mesh for the blendshapes: "full" (default, Face Landmarker's mesh in the
Holistic bundle) or "lite" (stock Holistic, slightly cheaper, less accurate blendshapes; detector.py).
--no-face: no blendshapes. --no-fingers: no finger bones.
--face-rate: blendshapes go out on their own at this rate (default 60 Hz), glided between detector frames (face.py);
0 sends them with each detector frame (for receivers that interpolate themselves). --no-eye-filter: raw eye shapes
(no resting-level, look-down and left/right correction, mtcore.face_filter). --dump-landmarks keeps the raw values.

Standalone: runs until Ctrl+C or the end of the video; log lines go to stderr.

--stdio (driven by a parent such as vspark's Learned Tracking behavior): JSON lines.
  stdin (commands):  {"cmd": "avatar", "vrm_path": "/abs/avatar.vrm"}   avatar rest pose and retargeting from its file
                     {"cmd": "avatar", "rest_positions": {vrmBone: [x, y, z]} | null}  (null: default)
                     {"cmd": "camera", "index": n}      switch camera
                     {"cmd": "cameras"}                 list cameras
                     {"cmd": "stop"}                    exit (closing stdin does the same)
  stdout (events):   {"event": "ready", "model": ..., "bones": [...], "face": "full" | "lite" | null,
                                      "fingers": bool, "legs": bool, "merger": bool, "retarget": bool}
                     {"event": "status", "fps", "detect_ms", "model_ms", "latency_ms", "present"}  (about 1/s)
                     {"event": "cameras", "cameras": [{"index", "name"}]}
                     {"event": "error", "message": ...}
                     {"event": "stopped", "reason": ...}
stdout carries nothing else; MediaPipe's own logging goes to stderr.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

from mtcore.paths import DATA

from .detector import Detector, model_path, prepare
from .face import EyeFilter, FaceStream
from .hands import solve_hands
from .lifter import LiveLifter
from mtcore.merged import LiveMerged
from .sources import CameraSource, VideoSource, list_cameras
from .vmc import VmcSender

DEFAULT_BUNDLE = DATA / "m1" / "runs" / "m1-v10" / "live"
DEFAULT_MERGER = DATA / "merger" / "runs" / "mg3" / "live"
LEG_WORDS = ("UpperLeg", "LowerLeg", "Foot", "Toes")
FINGER_WORDS = ("Thumb", "Index", "Middle", "Ring", "Little")


class LifterModel:
    """--no-merger: the M1 lifter with hands.py's fingers, in LiveMerged's shape."""

    def __init__(self, bundle, head3d, fingers):
        self.lifter, self.fingers = LiveLifter(bundle, head3d=head3d), fingers
        self.meta, self.bones = self.lifter.meta, self.lifter.bones
        self.name = self.meta["source"]["run"]

    @property
    def rest(self):
        return self.lifter.rest

    def set_avatar(self, rest_positions=None):
        self.lifter.set_avatar(rest_positions)

    def set_avatar_vrm(self, path):
        self.lifter.set_avatar_vrm(path)

    def reset(self):
        self.lifter.reset()

    def step(self, t, obs):
        bones, conf, present = self.lifter.step(t, obs)
        if self.fingers:
            bones.update(solve_hands(obs))  # only hands seen this frame; receivers keep the last value
        return bones, conf, present, None, None


class MergerModel(LiveMerged):
    def __init__(self, bundle, head3d, fingers, legs):
        super().__init__(bundle, head3d=head3d)
        self.name = f"{self.meta['source']['merger']} ({self.meta['source']['body']}, {self.meta['source']['hands']})"
        self.bones = [b for b in self.bones if (fingers or not any(w in b for w in FINGER_WORDS))
                      and (legs or not any(w in b for w in LEG_WORDS))]


def load_model(a):
    """--merger / --no-merger -> the model (see the module docstring)."""
    merger = a.merger or (DEFAULT_MERGER if (DEFAULT_MERGER / "merger.json").exists() else None)
    if a.no_merger or merger is None:
        if not a.no_merger:
            log(f"no merger bundle at {DEFAULT_MERGER}: M1 lifter only")
        return LifterModel(a.bundle, not a.no_head3d, not a.no_fingers)
    return MergerModel(merger, not a.no_head3d, not a.no_fingers, not a.no_legs)


class Events:
    def __init__(self, enabled):
        self.enabled, self.lock = enabled, threading.Lock()

    def __call__(self, event, **kw):
        if not self.enabled:
            return
        with self.lock:
            sys.stdout.write(json.dumps({"event": event, **kw}) + "\n")
            sys.stdout.flush()


def log(*a):
    print("[mtservice]", *a, file=sys.stderr, flush=True)


def stdin_commands(q):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            q.put(json.loads(line))
        except json.JSONDecodeError as e:
            q.put({"cmd": "_bad", "error": str(e)})
    q.put({"cmd": "stop", "reason": "stdin closed"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None)
    src.add_argument("--video", type=Path)
    ap.add_argument("--realtime", action="store_true", help="video: pace to the wall clock")
    ap.add_argument("--vmc", default="127.0.0.1:39539", help="HOST:PORT, or 'off'")
    ap.add_argument("--merger", type=Path, help=f"merger bundle (default {DEFAULT_MERGER} if present)")
    ap.add_argument("--no-merger", action="store_true", help="M1 lifter alone (--bundle) with hands.py's fingers")
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="M1 lifter bundle for --no-merger")
    ap.add_argument("--no-legs", action="store_true", help="don't send leg bones")
    ap.add_argument("--task", type=Path, help="Holistic .task file (default: pinned model, downloaded once)")
    ap.add_argument("--stdio", action="store_true")
    ap.add_argument("--dump-poses", type=Path)
    ap.add_argument("--dump-landmarks", type=Path)
    ap.add_argument("--avatar-vrm", type=Path,
                    help="avatar .vrm: the bones are computed for its rest pose and retargeted to its proportions")
    ap.add_argument("--no-retarget", action="store_true",
                    help="plain rotation copy onto the avatar (no relational retargeting)")
    ap.add_argument("--face-mesh", choices=["full", "lite"], default="full",
                    help="face mesh for the blendshapes (default full; ignored with --task)")
    ap.add_argument("--no-face", action="store_true", help="don't compute or send face blendshapes")
    ap.add_argument("--face-rate", type=float, default=60.0,
                    help="send blendshapes on their own at this rate, glided (default 60; 0: with each detector frame)")
    ap.add_argument("--no-eye-filter", action="store_true", help="raw eye blendshapes (no resting/look-down/symmetry fix)")
    ap.add_argument("--no-fingers", action="store_true", help="don't compute or send finger bones")
    ap.add_argument("--no-head3d", action="store_true")
    ap.add_argument("--list-cameras", action="store_true")
    a = ap.parse_args(argv)
    if a.list_cameras:
        print(json.dumps(list_cameras()))
        return 0
    emit = Events(a.stdio)
    try:
        reason = run(a, emit)
        code = 0
    except Exception as e:  # report, then fail
        emit("error", message=f"{type(e).__name__}: {e}")
        log(f"error: {type(e).__name__}: {e}")
        reason, code = "error", 1
    emit("stopped", reason=reason)
    log(f"stopped: {reason}")
    return code


def set_avatar_vrm(lifter, path, emit):
    """Avatar from a VRM file; a retargeting failure is reported and leaves the rotation copy."""
    t0 = time.monotonic()
    lifter.set_avatar_vrm(path)
    err = getattr(lifter, "retarget_error", None)
    if err:
        emit("error", message=f"retarget: {err}")
        log(f"retarget off: {err}")
    elif getattr(lifter, "retarget", None) is not None:
        log(f"retargeting to {Path(path).name} ({(time.monotonic() - t0) * 1000:.0f} ms to set up)")


def open_source(a, camera):
    if a.video:
        return VideoSource(a.video, realtime=a.realtime)
    return CameraSource(camera)


def run(a, emit):
    """Main loop; returns the stop reason."""
    commands = queue.Queue()
    if a.stdio:
        threading.Thread(target=stdin_commands, args=(commands,), daemon=True).start()
    lifter = load_model(a)
    if a.no_retarget and isinstance(lifter, LiveMerged):
        lifter.retarget_enabled = False
    if a.avatar_vrm:
        set_avatar_vrm(lifter, a.avatar_vrm, emit)
    detector = Detector(model_path(a.task, face_mesh=a.face_mesh, log=log), blendshapes=not a.no_face)
    vmc = None
    if a.vmc != "off":
        host, port = a.vmc.rsplit(":", 1)
        vmc = VmcSender(host, int(port))
        vmc.set_rest(lifter.rest)
    camera = a.camera if a.camera is not None else 0
    source = open_source(a, camera)
    eyes = None if a.no_face or a.no_eye_filter else EyeFilter()
    paced = not a.video or a.realtime  # the face stream glides on the wall clock
    face = FaceStream(vmc, a.face_rate) if vmc and not a.no_face and a.face_rate > 0 and paced else None
    dump_p = open(a.dump_poses, "w") if a.dump_poses else None
    dump_l = open(a.dump_landmarks, "w") if a.dump_landmarks else None
    merged = isinstance(lifter, LiveMerged)
    emit("ready", model=lifter.meta["source"], bones=lifter.bones, face=None if a.no_face else a.face_mesh,
         fingers=not a.no_fingers, legs=merged and not a.no_legs, merger=merged,
         retarget=merged and lifter.retarget is not None,
         source={"video": str(a.video)} if a.video else {"camera": camera})
    log(f"running: model {lifter.name}, " + (f"video {a.video}" if a.video else f"camera {camera}")
        + (f", VMC -> {a.vmc}" if vmc else ""))
    stats = {"n": 0, "det": 0.0, "mod": 0.0, "lat": 0.0, "present": 0}
    last_status = time.monotonic()
    try:
        it = iter(source)
        while True:
            restart = False
            while not commands.empty():
                c = commands.get()
                cmd = c.get("cmd")
                if cmd == "stop":
                    return c.get("reason", "stop command")
                if cmd == "avatar":
                    try:
                        if c.get("vrm_path"):
                            set_avatar_vrm(lifter, c["vrm_path"], emit)
                        else:
                            lifter.set_avatar(c.get("rest_positions"))
                    except Exception as e:  # keep running with the previous avatar
                        emit("error", message=f"avatar: {type(e).__name__}: {e}")
                    if vmc:
                        vmc.set_rest(lifter.rest)
                elif cmd == "cameras":
                    emit("cameras", cameras=list_cameras())
                elif cmd == "camera" and not a.video:
                    camera = int(c["index"])
                    restart = True
                elif cmd == "_bad":
                    emit("error", message=f"bad command: {c['error']}")
                else:
                    emit("error", message=f"unknown command: {cmd}")
            if restart:
                source.close()
                source = open_source(a, camera)
                it = iter(source)
                lifter.reset()
                if eyes:
                    eyes.reset()
                if face:
                    face.reset()
            try:
                t, frame = next(it)
            except StopIteration:
                return "end of video"
            wall0 = time.monotonic()
            obs = detector.detect(prepare(frame), t * 1000 + 1)
            wall1 = time.monotonic()
            bones, conf, present, root, up = lifter.step(t, obs)
            shapes = obs.get("blendshapes")
            if shapes and eyes:
                shapes = eyes.step(t, shapes)
            wall2 = time.monotonic()
            if face:
                face.push(shapes)
            if vmc:
                vmc.send(t, bones, conf, present, None if face else shapes, root=root, up=up)
            if dump_p:
                rec = {"t": round(t * 1000, 1), "bones": {b: [round(float(v), 6) for v in q] for b, q in bones.items()},
                       "conf": {j: round(c, 3) for j, c in conf.items()}}
                if root is not None:  # camera frame: metres, unit vector
                    rec.update(root=[round(float(v), 4) for v in root], up=[round(float(v), 4) for v in up])
                dump_p.write(json.dumps(rec) + "\n")
            if dump_l:
                f = {"t": round(t * 1000, 1)}
                for k, v in obs.items():
                    if k == "blendshapes":
                        f[k] = {n: round(float(x), 4) for n, x in v.items()}
                    else:
                        f[k] = [[round(float(x), 5) for x in row] for row in v]
                dump_l.write(json.dumps(f) + "\n")
            stats["n"] += 1
            stats["det"] += wall1 - wall0
            stats["mod"] += wall2 - wall1
            if not a.video:
                stats["lat"] += wall2 - (source.t0 + t)
            stats["present"] += present
            now = time.monotonic()
            if now - last_status >= 1.0:
                n = stats["n"]
                emit("status", fps=round(n / (now - last_status), 1), detect_ms=round(1000 * stats["det"] / n, 1),
                     model_ms=round(1000 * stats["mod"] / n, 2), latency_ms=round(1000 * stats["lat"] / n, 1) if not a.video else None,
                     present=round(stats["present"] / n, 2))
                stats = {"n": 0, "det": 0.0, "mod": 0.0, "lat": 0.0, "present": 0}
                last_status = now
    finally:
        if face:
            face.close()
        source.close()
        detector.close()
        if vmc:
            vmc.close()
        for f in (dump_p, dump_l):
            if f:
                f.close()


if __name__ == "__main__":
    sys.exit(main())
