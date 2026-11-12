"""mtservice: camera or video -> MediaPipe Holistic -> M1 lifter -> VMC.

    mtservice [--camera N | --video FILE [--realtime]] [--vmc HOST:PORT] [--bundle DIR] [--task FILE]
              [--avatar-vrm FILE] [--face-mesh full|lite] [--no-face] [--stdio] [--dump-poses FILE]
              [--dump-landmarks FILE] [--no-head3d] [--list-cameras]

--face-mesh: MediaPipe face mesh for the blendshapes: "full" (default, Face Landmarker's mesh in the
Holistic bundle) or "lite" (stock Holistic, slightly cheaper, less accurate blendshapes; detector.py).
--no-face: no blendshapes.

Standalone: runs until Ctrl+C or the end of the video; log lines go to stderr.

--stdio (driven by a parent such as vspark's Learned Tracking behavior): JSON lines.
  stdin (commands):  {"cmd": "avatar", "vrm_path": "/abs/avatar.vrm"}   avatar rest pose from its file
                     {"cmd": "avatar", "rest_positions": {vrmBone: [x, y, z]} | null}  (null: default)
                     {"cmd": "camera", "index": n}      switch camera
                     {"cmd": "cameras"}                 list cameras
                     {"cmd": "stop"}                    exit (closing stdin does the same)
  stdout (events):   {"event": "ready", "model": ..., "bones": [...], "face": "full" | "lite" | null}
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
from .lifter import LiveLifter
from .sources import CameraSource, VideoSource, list_cameras
from .vmc import VmcSender

DEFAULT_BUNDLE = DATA / "m1" / "runs" / "m1-v10" / "live"


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
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--task", type=Path, help="Holistic .task file (default: pinned model, downloaded once)")
    ap.add_argument("--stdio", action="store_true")
    ap.add_argument("--dump-poses", type=Path)
    ap.add_argument("--dump-landmarks", type=Path)
    ap.add_argument("--avatar-vrm", type=Path, help="avatar .vrm whose rest pose the bones are computed for")
    ap.add_argument("--face-mesh", choices=["full", "lite"], default="full",
                    help="face mesh for the blendshapes (default full; ignored with --task)")
    ap.add_argument("--no-face", action="store_true", help="don't compute or send face blendshapes")
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


def open_source(a, camera):
    if a.video:
        return VideoSource(a.video, realtime=a.realtime)
    return CameraSource(camera)


def run(a, emit):
    """Main loop; returns the stop reason."""
    commands = queue.Queue()
    if a.stdio:
        threading.Thread(target=stdin_commands, args=(commands,), daemon=True).start()
    lifter = LiveLifter(a.bundle, head3d=not a.no_head3d)
    if a.avatar_vrm:
        lifter.set_avatar_vrm(a.avatar_vrm)
    detector = Detector(model_path(a.task, face_mesh=a.face_mesh, log=log), blendshapes=not a.no_face)
    vmc = None
    if a.vmc != "off":
        host, port = a.vmc.rsplit(":", 1)
        vmc = VmcSender(host, int(port))
        vmc.set_rest(lifter.rest)
    camera = a.camera if a.camera is not None else 0
    source = open_source(a, camera)
    dump_p = open(a.dump_poses, "w") if a.dump_poses else None
    dump_l = open(a.dump_landmarks, "w") if a.dump_landmarks else None
    emit("ready", model=lifter.meta["source"], bones=lifter.bones, face=None if a.no_face else a.face_mesh,
         source={"video": str(a.video)} if a.video else {"camera": camera})
    log(f"running: model {lifter.meta['source']['run']}, " + (f"video {a.video}" if a.video else f"camera {camera}")
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
                            lifter.set_avatar_vrm(c["vrm_path"])
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
            try:
                t, frame = next(it)
            except StopIteration:
                return "end of video"
            wall0 = time.monotonic()
            obs = detector.detect(prepare(frame), t * 1000 + 1)
            wall1 = time.monotonic()
            bones, conf, present = lifter.step(t, obs)
            wall2 = time.monotonic()
            if vmc:
                vmc.send(t, bones, conf, present, obs.get("blendshapes"))
            if dump_p:
                dump_p.write(json.dumps({"t": round(t * 1000, 1), "bones": {b: [round(float(v), 6) for v in q] for b, q in bones.items()},
                                         "conf": {j: round(c, 3) for j, c in conf.items()}}) + "\n")
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
        source.close()
        detector.close()
        if vmc:
            vmc.close()
        for f in (dump_p, dump_l):
            if f:
                f.close()


if __name__ == "__main__":
    sys.exit(main())
