"""Camera preview for a parent process (`--stdio`): the frame the detector saw and/or its 2D landmarks.

Off until the stdin command {"cmd": "preview", "image": bool, "landmarks": bool} asks for it; then one `preview`
event per processed frame (the runtime's rate, about 10 per second):
  {"event": "preview", "t": ms, "width": 320, "height": 240, "mirrored": true,
   "image": base64 JPEG (with image),
   "landmarks": {"pose": [[x, y, visibility] x33], "leftHand": [[x, y] x21], "rightHand": ..., "face": [[x, y] x478]}
                (with landmarks; streams not detected this frame are left out)}
Coordinates are normalized to the image (0..1, x right, y down) of the mirrored frame, so they line up with the
image. left/right are MediaPipe's on the mirrored frame (as everywhere in the runtime, detector.py). A 320x240
JPEG at quality 70 is about 10-15 KB and takes about 1 ms to encode.
"""

import base64

import cv2

QUALITY = 70
STREAMS = {"pose": "pose2d", "leftHand": "leftHand2d", "rightHand": "rightHand2d", "face": "face"}


class Preview:
    def __init__(self):
        self.image = False
        self.landmarks = False

    @property
    def on(self):
        return self.image or self.landmarks

    def configure(self, image=False, landmarks=False):
        self.image, self.landmarks = bool(image), bool(landmarks)

    def event(self, t_ms, rgb, obs):
        """Fields of the `preview` event for one frame (rgb: the detector's mirrored RGB input)."""
        h, w = rgb.shape[:2]
        ev = {"t": round(t_ms, 1), "width": w, "height": h, "mirrored": True}
        if self.image:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if ok:
                ev["image"] = base64.b64encode(buf.tobytes()).decode("ascii")
        if self.landmarks:
            lm = {}
            for name, key in STREAMS.items():
                v = obs.get(key)
                if v is None:
                    continue
                cols = [0, 1, 3] if name == "pose" else [0, 1]
                lm[name] = [[round(float(r[c]), 4) for c in cols] for r in v]
            ev["landmarks"] = lm
        return ev
