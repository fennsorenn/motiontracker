"""preview.py: the preview event's image and landmarks."""

import base64

import cv2
import numpy as np

from mtservice.preview import Preview


def test_preview_fields():
    rgb = np.zeros((240, 320, 3), np.uint8)
    rgb[:, :160] = (255, 0, 0)
    obs = {"pose2d": np.random.rand(33, 4), "leftHand2d": np.random.rand(21, 3), "pose": np.random.rand(33, 4)}
    p = Preview()
    assert not p.on
    p.configure(image=True)
    ev = p.event(1234.56, rgb, obs)
    assert ev["width"] == 320 and ev["height"] == 240 and "landmarks" not in ev
    img = cv2.imdecode(np.frombuffer(base64.b64decode(ev["image"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 320, 3) and img[120, 40, 2] > 200  # red stays red (BGR on decode)
    p.configure(landmarks=True)
    ev = p.event(0, rgb, obs)
    assert "image" not in ev and set(ev["landmarks"]) == {"pose", "leftHand"}
    assert len(ev["landmarks"]["pose"][0]) == 3 and len(ev["landmarks"]["leftHand"][0]) == 2
    assert abs(ev["landmarks"]["pose"][5][2] - obs["pose2d"][5][3]) < 1e-4  # visibility
