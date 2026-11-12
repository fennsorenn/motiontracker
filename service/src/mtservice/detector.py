"""MediaPipe Holistic Landmarker, set up like the recorder (tools/recorder): same model, VIDEO mode, same
confidence options, mirrored 320x240 input. Returns the observations the lifter reads, plus the 52 ARKit
face blendshapes from MediaPipe's blendshape model (shipped inside the Holistic bundle).

Face mesh (`face_mesh`): Holistic ships the lite Face Mesh V2; the Face Landmarker bundle ships the full
one, same 478-point layout. "full" (default) runs Holistic with the full mesh swapped into its bundle:
blendshapes then agree with Face Landmarker's much more closely (median correlation 0.89 vs 0.69 on the
recorded session) for about 0.2 ms/frame, and body and hand outputs are bit-identical. "lite" is the
stock bundle (PLAN.md §7.5).

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
import zipfile

from mediapipe.framework.formats import classification_pb2, landmark_pb2
from mediapipe.python import packet_creator, packet_getter
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import holistic_landmarker as hl

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/"
             "float16/1/holistic_landmarker.task")
MODEL_MD5_BASE64 = "3ZywDp2ZM9rHmI2M40e6ew=="
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
            "float16/1/face_landmarker.task")
FACE_MD5_BASE64 = "sOcnSQehZEQE/vZrKN1thQ=="
FACE_MESH_ENTRY = "face_landmarks_detector.tflite"  # same entry name in both bundles
FACE_MESHES = ("full", "lite")
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


def _download(url, md5_base64, path, log):
    if path.exists():
        return path
    log(f"downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(url, timeout=60).read()
    md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    if md5 != md5_base64:
        raise RuntimeError(f"{url}: checksum mismatch: {md5} != {md5_base64}")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def model_path(path=None, face_mesh="full", log=print):
    """The Holistic .task file: the given path, else the pinned model downloaded once into the cache; with
    face_mesh "full", a copy of it with the Face Landmarker bundle's full face mesh (both pinned, built once)."""
    if path:
        return Path(path)
    if face_mesh not in FACE_MESHES:
        raise ValueError(f"face_mesh must be one of {FACE_MESHES}, not {face_mesh!r}")
    holistic = _download(MODEL_URL, MODEL_MD5_BASE64, cache_dir() / "holistic_landmarker-float16-1.task", log)
    if face_mesh == "lite":
        return holistic
    out = cache_dir() / "holistic_landmarker-float16-1+face_landmarker-float16-1-mesh.task"
    if out.exists():
        return out
    face = _download(FACE_URL, FACE_MD5_BASE64, cache_dir() / "face_landmarker-float16-1.task", log)
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(holistic) as src, zipfile.ZipFile(face) as fl, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            name = info.filename
            dst.writestr(name, fl.read(name) if name == FACE_MESH_ENTRY else src.read(name))
    tmp.rename(out)
    return out


def prepare(frame_bgr):
    """Camera frame (BGR, any size) -> mirrored 320x240 RGB, as the recorder draws it."""
    img = cv2.resize(cv2.flip(frame_bgr, 1), (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Detector:
    def __init__(self, task_path, blendshapes=True):
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)), running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5, output_face_blendshapes=blendshapes)
        self.lm = vision.HolisticLandmarker.create_from_options(opts)
        self.blendshapes = blendshapes
        self.last_ts = -1

    def close(self):
        self.lm.close()

    def detect(self, rgb, ts_ms):
        """Mirrored 320x240 RGB frame, timestamp (ms, strictly increasing) -> dict of landmark arrays:
        pose (33, 4) world xyz + image visibility, pose2d (33, 4) image xyz + visibility, face (478, 3),
        hands as (21, 3); blendshapes {ARKit name: 0..1}; absent streams are missing from the dict.
        Left/right everywhere is MediaPipe's on the mirrored frame, so the avatar mirrors the person: the
        pose's "left" shoulder is the person's right (image right in every frame of the recorded
        session), and the *Left blendshapes follow the image's right side too (measured: eyeLook shapes
        track gaze toward image right on mirrored and unmirrored frames alike). Face and body agree, so
        the blendshapes are passed on unswapped, as vspark's MediaPipe Tracking does."""
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
        if self.blendshapes:
            p = packets.get(hl._FACE_BLENDSHAPES_STREAM_NAME)
            if p is not None and not p.is_empty():
                msg = classification_pb2.ClassificationList()
                msg.MergeFrom(packet_getter.get_proto(p))
                shapes = {c.label: c.score for c in msg.classification if c.label != "_neutral"}
                if shapes:
                    out["blendshapes"] = shapes
        if "pose" in out and "pose2d" in out:
            out["pose"][:, 3] = out["pose2d"][:, 3]  # vspark/recorder: world points, image visibility
        return out
