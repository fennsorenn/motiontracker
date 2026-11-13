"""MediaPipe Hand Landmarker (standalone, own palm detector) over rendered sequences -> <seq>/landmarks-hl.jsonl in
the recorder's frame format, for comparing detectors on renders (plan m4-hand-model).

    uv run --project service python tools/render/handlandmarker.py data/m4/renders/<set>

Per frame: leftHand / rightHand (world landmarks), leftHand2d / rightHand2d (image landmarks) by the detector's
handedness label (at most one per label, the higher score), and leftHandScore / rightHandScore, its handedness
score (the detector's confidence input of the pair hand model, PLAN.md section 5.6 v3). Frame times follow the
sequence's Holistic landmarks.jsonl. Model: hand_landmarker float16 v1 (Apache 2.0, docs/LICENSES-REGISTER.md).
"""

import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from mtservice.detector import cache_dir, prepare
from mtservice.sources import VideoSource


def main():
    for d in sorted(Path(sys.argv[1]).glob("seq*")):
        done = d / "landmarks-hl.jsonl"
        if done.exists() and "HandScore" in done.read_text():  # resumable: finished sequences (with scores) skipped
            print(d.name, "already done", flush=True)
            continue
        hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cache_dir() / "hand_landmarker-float16-1.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))
        ref = [json.loads(line)["t"] for line in open(d / "landmarks.jsonl")]
        tmp = d / "landmarks-hl.tmp.jsonl"
        with open(tmp, "w") as out:
            for (t, f), tr in zip(VideoSource(d / "video.webm"), ref):
                r = hl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(prepare(f))),
                                        int(t * 1000) + 1)
                fr = {"t": tr}
                best = {}
                for h, lm, w in zip(r.handedness, r.hand_landmarks, r.hand_world_landmarks):
                    k = "leftHand" if h[0].category_name == "Left" else "rightHand"
                    if k not in best or h[0].score > best[k][0]:
                        best[k] = (h[0].score, lm, w)
                for k, (score, lm, w) in best.items():
                    fr[k] = [[p.x, p.y, p.z] for p in w]
                    fr[k + "2d"] = [[p.x, p.y, p.z] for p in lm]
                    fr[k + "Score"] = round(float(score), 4)
                out.write(json.dumps(fr) + "\n")
        hl.close()
        tmp.replace(done)
        print(d.name, "done", flush=True)


if __name__ == "__main__":
    main()
