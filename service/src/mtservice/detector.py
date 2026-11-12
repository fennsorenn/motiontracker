"""MediaPipe Holistic Landmarker, set up like the recorder (tools/recorder): same model, VIDEO mode, same
confidence options, mirrored 320x240 input. Returns the observations the lifter reads.

mediapipe 0.10.18's Python HolisticLandmarker.detect_for_video aborts the process on an empty output
packet (e.g. no hand) and drops everything when the face is missing; `detect` reads each stream itself
(spikes/mp-python-parity).
"""

import base64
import hashlib
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator, packet_getter
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import holistic_landmarker as hl

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/"
             "float16/1/holistic_landmarker.task")
MODEL_MD5_BASE64 = "3ZywDp2ZM9rHmI2M40e6ew=="
W, H = 320, 240

STREAMS = {
    "face": (hl._FACE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "pose2d": (hl._POSE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "pose": (hl._POSE_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
    "leftHand2d": (hl._LEFT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "leftHand": (hl._LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
    "rightHand2d": (hl._RIGHT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "rightHand": (hl._RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
}


def cache_dir():
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "motiontracker"


def model_path(path=None, log=print):
    """The Holistic .task file: the given path, or the pinned model downloaded once into the cache."""
    if path:
        return Path(path)
    p = cache_dir() / "holistic_landmarker-float16-1.task"
    if not p.exists():
        log(f"downloading {MODEL_URL}")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(MODEL_URL, timeout=60).read()
        md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
        if md5 != MODEL_MD5_BASE64:
            raise RuntimeError(f"model checksum mismatch: {md5} != {MODEL_MD5_BASE64}")
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(p)
    return p


def prepare(frame_bgr):
    """Camera frame (BGR, any size) -> mirrored 320x240 RGB, as the recorder draws it."""
    img = cv2.resize(cv2.flip(frame_bgr, 1), (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Detector:
    def __init__(self, task_path):
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)), running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5)
        self.lm = vision.HolisticLandmarker.create_from_options(opts)
        self.last_ts = -1

    def close(self):
        self.lm.close()

    def detect(self, rgb, ts_ms):
        """Mirrored 320x240 RGB frame, timestamp (ms, strictly increasing) -> dict of landmark arrays:
        pose (33, 4) world xyz + image visibility, pose2d (33, 4) image xyz + visibility, face (478, 3),
        hands as (21, 3); absent streams are missing from the dict."""
        ts_ms = max(int(ts_ms), self.last_ts + 1)
        self.last_ts = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        packets = self.lm._process_video_data({
            hl._IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(ts_ms * hl._MICRO_SECONDS_PER_MILLISECOND)})
        out = {}
        for key, (stream, cls) in STREAMS.items():
            p = packets.get(stream)
            if p is None or p.is_empty():
                continue
            msg = cls()
            msg.MergeFrom(packet_getter.get_proto(p))
            if not msg.landmark:
                continue
            if key.startswith("pose"):
                out[key] = np.array([[l.x, l.y, l.z, l.visibility] for l in msg.landmark])
            else:
                out[key] = np.array([[l.x, l.y, l.z] for l in msg.landmark])
        if "pose" in out and "pose2d" in out:
            out["pose"][:, 3] = out["pose2d"][:, 3]  # vspark/recorder: world points, image visibility
        return out
