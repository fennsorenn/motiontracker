"""Frame sources. Each yields (t seconds, BGR frame) at the recorder's 10 FPS rhythm.

- CameraSource: a reader thread keeps only the newest camera frame (no buffered lag); every 1/fps
  seconds the newest frame is taken, like the recorder's throttled capture loop. t = capture time
  since start (monotonic).
- VideoSource: the recorder's stepped replay (ts = k / fps + 1 ms, latest frame at or before ts);
  deterministic. realtime=True paces it to the wall clock (for driving vspark from a file).
"""

import glob
import os
import sys
import threading
import time
from pathlib import Path

import cv2


def list_cameras():
    """[{'index': n, 'name': str}] of capture devices. Linux: from sysfs (metadata nodes skipped)."""
    out = []
    if sys.platform.startswith("linux"):
        for d in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
            try:
                if Path(d, "index").read_text().strip() != "0":
                    continue
                out.append({"index": int(d.rsplit("video", 1)[1]), "name": Path(d, "name").read_text().strip()})
            except OSError:
                continue
        return out
    for i in range(6):  # elsewhere: probe
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            out.append({"index": i, "name": f"camera {i}"})
        cap.release()
    return out


class CameraSource:
    def __init__(self, index, fps=10.0, width=640, height=480):
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.interval = 1.0 / fps
        self.t0 = time.monotonic()
        self.latest, self.seq, self.running = None, 0, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.latest = (time.monotonic() - self.t0, frame)
                self.seq += 1
                self.cond.notify_all()

    def __iter__(self):
        last_seq, next_at = 0, time.monotonic()
        while self.running:
            wait = next_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout=1.0)
                if not self.running:
                    return
                if self.seq == last_seq:
                    continue  # camera stalled; keep waiting
                last_seq, item = self.seq, self.latest
            next_at = time.monotonic() + self.interval
            yield item

    def close(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=2)
        self.cap.release()


class VideoSource:
    def __init__(self, path, fps=10.0, realtime=False):
        if not os.path.exists(path):
            raise RuntimeError(f"no such video: {path}")
        self.path, self.step, self.realtime, self.running = path, 1.0 / fps, realtime, True

    def __iter__(self):
        cap = cv2.VideoCapture(str(self.path))
        try:
            ok, frame = cap.read()
            cur = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, frame) if ok else None
            nxt = None
            start = time.monotonic()
            k = 0
            while self.running and cur is not None:
                ts = k * self.step + 0.001
                while True:  # advance to the latest frame at or before ts
                    if nxt is None:
                        ok, f = cap.read()
                        nxt = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, f) if ok else False
                    if nxt and nxt[0] <= ts:
                        cur, nxt = nxt, None
                    else:
                        break
                if nxt is False and ts >= cur[0] + 0.001:
                    return  # past the last frame (as spikes/mp-python-parity)
                if self.realtime:
                    wait = start + ts - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
                yield ts, cur[1]
                k += 1
        finally:
            cap.release()

    def close(self):
        self.running = False
