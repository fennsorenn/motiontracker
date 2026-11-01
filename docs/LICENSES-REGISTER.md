# License register

Every asset, dataset, model, recording and third-party code used in this repo. Check PLAN.md §2.3
and §8 before adding anything. The project's own license is still open (PLAN.md §9 #9).

| Item | Source | License | Used for | Notes |
|---|---|---|---|---|
| vspark solver code (`baseline/vendor/`) | Owner's own repo `vspark` (private) | **None declared** (owner's copyright) | M0 baseline only | Owner must relicense before any public release, or keep `baseline/` out of it. |
| vspark webcam arm clips (`baseline/test/fixtures/webcam_arm_clips.json`) | vspark test fixtures; MediaPipe Holistic output from the owner's webcam | Owner's own recording | Baseline equivalence test | Landmarks of the owner, no images. Owner decides whether it ships publicly. |
| three.js | https://github.com/mrdoob/three.js (npm `three` ~0.168) | MIT | Quaternion math in vendored frontend filters | |
| MediaPipe Holistic Landmarker (indirectly) | Google, via vspark's camera code | Apache 2.0 (per model card; verify) | Produced the recorded landmarks | Not bundled here. |
