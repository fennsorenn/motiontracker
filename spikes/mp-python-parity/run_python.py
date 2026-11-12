"""Python MediaPipe Holistic over a video, frame selection and output format as the recorder's stepped replay.

    uv run python run_python.py <video> <holistic_landmarker.task> <out landmarks.jsonl> [--resize linear|area]

Recorder (tools/recorder/recorder.js, steppedReplay + processFrame + toFrame): for k = 0, 1, ...
seek to ts = k * 0.1 s + 1 ms, take the frame shown there (latest pts <= ts), mirror it, draw it at
320x240, detectForVideo(frame, ts * 1000 + 1), record t = round(ts * 1000).
"""

import argparse
import json

from types import SimpleNamespace

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator, packet_getter
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import holistic_landmarker as hl

W, H, STEP = 320, 240, 0.1

# mediapipe 0.10.18's Python HolisticLandmarker.detect_for_video reads every output stream with
# get_proto, which aborts the process on an empty packet (e.g. no hand), and returns an empty result
# for everything when the face is missing. Read each stream on its own instead; the graph is the same.
STREAMS = {
    "face_landmarks": (hl._FACE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "pose_landmarks": (hl._POSE_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "pose_world_landmarks": (hl._POSE_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
    "left_hand_landmarks": (hl._LEFT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "left_hand_world_landmarks": (hl._LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
    "right_hand_landmarks": (hl._RIGHT_HAND_LANDMARKS_STREAM_NAME, landmark_pb2.NormalizedLandmarkList),
    "right_hand_world_landmarks": (hl._RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME, landmark_pb2.LandmarkList),
}


def detect_for_video(lm, image, ts_ms):
    packets = lm._process_video_data({
        hl._IMAGE_IN_STREAM_NAME: packet_creator.create_image(image).at(ts_ms * hl._MICRO_SECONDS_PER_MILLISECOND)})
    out = {}
    for field, (stream, cls) in STREAMS.items():
        p = packets.get(stream)
        if p is None or p.is_empty():
            out[field] = []
            continue
        msg = cls()
        msg.MergeFrom(packet_getter.get_proto(p))
        out[field] = list(msg.landmark)
    return SimpleNamespace(**out)


def r5(v):
    return round(v * 1e5) / 1e5


def first(x):
    """Python results are a flat landmark list or a list of lists, depending on the field/version."""
    if not x:
        return None
    return x[0] if isinstance(x[0], list) else x


def pack(pts, vis):
    return [[r5(p.x), r5(p.y), r5(p.z)] + ([r5(p.visibility or 0.0)] if vis else []) for p in pts]


def to_frame(r, t):
    f = {"t": t}
    face, pose, posew = first(r.face_landmarks), first(r.pose_landmarks), first(r.pose_world_landmarks)
    if face:
        f["face"] = pack(face, False)
    if posew:
        f["pose"] = [[r5(p.x), r5(p.y), r5(p.z), r5((pose[i].visibility if pose else p.visibility) or 0.0)]
                     for i, p in enumerate(posew)]
    elif pose:
        f["pose"] = pack(pose, True)
    for side in ("left", "right"):
        world, img = first(getattr(r, f"{side}_hand_world_landmarks")), first(getattr(r, f"{side}_hand_landmarks"))
        if world or img:
            f[f"{side}Hand"] = pack(world or img, False)
        if img:
            f[f"{side}Hand2d"] = pack(img, False)
    if pose:
        f["pose2d"] = pack(pose, True)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--resize", choices=("linear", "area"), default="linear")
    a = ap.parse_args()
    opts = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=a.model), running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5, min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5, min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5)
    interp = cv2.INTER_LINEAR if a.resize == "linear" else cv2.INTER_AREA
    cap = cv2.VideoCapture(a.video)
    frames = []  # (pts seconds, frame)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, img))
    duration = frames[-1][0] + 1e-3
    n = 0
    with vision.HolisticLandmarker.create_from_options(opts) as lm, open(a.out, "w") as out:
        j = 0
        for k in range(10 ** 6):
            ts = k * STEP + 0.001
            if ts >= duration:
                break
            while j + 1 < len(frames) and frames[j + 1][0] <= ts:
                j += 1
            img = cv2.flip(frames[j][1], 1)
            img = cv2.resize(img, (W, H), interpolation=interp)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r = detect_for_video(lm, mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts * 1000 + 1))
            out.write(json.dumps(to_frame(r, round(ts * 1000))) + "\n")
            n += 1
    print(f"{len(frames)} video frames, {n} detections -> {a.out}")


if __name__ == "__main__":
    main()
