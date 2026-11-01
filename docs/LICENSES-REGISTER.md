# License register

Every asset, dataset, model, recording and third-party code used in this repo. Check PLAN.md §2.3
and §8 before adding anything. The project's own license is still open (PLAN.md §9 #9).

| Item | Source | License | Used for | Notes |
|---|---|---|---|---|
| vspark solver code (`baseline/vendor/`) | Owner's own repo `vspark` (private) | **None declared** (owner's copyright) | M0 baseline only | Owner must relicense before any public release, or keep `baseline/` out of it. |
| vspark webcam arm clips (`baseline/test/fixtures/webcam_arm_clips.json`) | vspark test fixtures; MediaPipe Holistic output from the owner's webcam | Owner's own recording | Baseline equivalence test | Landmarks of the owner, no images. Owner decides whether it ships publicly. |
| three.js | https://github.com/mrdoob/three.js (npm `three` ~0.168) | MIT | Quaternion math in vendored frontend filters | |
| MediaPipe Holistic Landmarker (indirectly) | Google, via vspark's camera code | Apache 2.0 (per model card; verify) | Produced the recorded landmarks | Not bundled here. |
| MediaPipe tasks-vision 0.10.17 (JS + WASM) | https://www.npmjs.com/package/@mediapipe/tasks-vision, loaded from jsDelivr by `tools/recorder` | Apache 2.0 | Landmark detection in the recorder | Not bundled. |
| MediaPipe Holistic Landmarker model, float16 v1 | storage.googleapis.com/mediapipe-models/holistic_landmarker/... (MD5 `3ZywDp2ZM9rHmI2M40e6ew==`) | Apache 2.0 (per model card; verify) | Recorder detection | Not bundled. |
| MediaPipe test image `male_full_height_hands.jpg` | storage.googleapis.com/mediapipe-assets/ | Apache 2.0 (MediaPipe test assets; verify) | Recorder headless test builds a video from it at run time | Not committed. |
| numpy | https://numpy.org | BSD-3-Clause | `eval/` metrics | |
| pytest (dev) | https://pytest.org | MIT | `eval/` tests | |
| vitest, tsx, typescript (dev) | npm | MIT / MIT / Apache 2.0 | `baseline/` tooling | |
| Owner's recorded sessions (`sessions/`) | Recorded with `tools/recorder` | Owner's own recordings | Evaluation | Gitignored; contain video of the owner. |
