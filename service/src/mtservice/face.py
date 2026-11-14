"""Face output of the runtime: eye post-processing per detector frame and a 60 Hz glided VMC stream
(`mtcore.face_filter`; reasons and measurements there).

The detector delivers blendshapes at its own rate (~10 Hz at the desk); sending them only then made the face step.
FaceStream keeps the newest processed values and a thread sends what `Glide` shows every 1/hz seconds, timed on
the wall clock (live camera, or a video paced with --realtime). Without a paced clock (a video run as fast as
possible) the runtime sends the processed values with each frame instead (`--face-rate 0` does the same live, for
receivers that interpolate themselves).
"""

import threading
import time

from mtcore.face_filter import EyeFilter, Glide


class FaceStream:
    def __init__(self, vmc, hz=60.0):
        self.vmc, self.hz = vmc, hz
        self.glide = Glide(hz)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def push(self, shapes):
        with self.lock:
            self.glide.push(time.monotonic(), shapes)

    def reset(self):
        with self.lock:
            self.glide.reset()

    def _run(self):
        tick = 1.0 / self.hz
        nxt = time.monotonic()
        while not self.stop.is_set():
            with self.lock:
                v = dict(self.glide.at(time.monotonic()))
            self.vmc.send_blend(v)
            nxt += tick
            time.sleep(max(0.0, nxt - time.monotonic()))
            if nxt < time.monotonic() - 1.0:  # fell far behind (suspended): resync
                nxt = time.monotonic()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)


__all__ = ["EyeFilter", "FaceStream"]
