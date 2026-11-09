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
| ai-edge-litert 2.3.0 (LiteRT Python interpreter) | https://pypi.org/project/ai-edge-litert/ | Apache 2.0 (PyPI classifier) | `spikes/mp-depth-probe`: runs MediaPipe's pose landmark model with intermediate tensors | Spike dependency only. |
| MediaPipe pose landmark model (from the Holistic bundle above) | extracted from `holistic_landmarker.task` (same MD5) into `data/spikes/mp-depth/model/` | Apache 2.0 (per model card; verify) | `spikes/mp-depth-probe`: activation probes | Not committed. A fine-tuned or edited derivative would need the model card license verified first. |
| MediaPipe test image `male_full_height_hands.jpg` | storage.googleapis.com/mediapipe-assets/ | Apache 2.0 (MediaPipe test assets; verify) | Recorder headless test builds a video from it at run time | Not committed. |
| numpy | https://numpy.org | BSD-3-Clause | `eval/` metrics | |
| pytest (dev) | https://pytest.org | MIT | `eval/` tests | |
| vitest, tsx, typescript (dev) | npm | MIT / MIT / Apache 2.0 | `baseline/` tooling | |
| Owner's recorded sessions (`sessions/`) | Recorded with `tools/recorder` | Owner's own recordings | Evaluation | Gitignored; contain video of the owner. |
| CMU Graphics Lab Motion Capture Database (`data/cmu/`) | http://mocap.cs.cmu.edu/allasfamc.zip (original ASF/AMC) | CMU terms: free for research; may be included in commercially sold products; may **not** resell the data directly, even in converted form | M1 training data | Gitignored, never commit or redistribute. Requested acknowledgment if results are published: "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217." |
| CMU → MHR fits and M1 training sets (`data/m1/`) | Derived from CMU by `train/` (`mttrain.prep_fits`) | CMU terms apply (converted data: no reselling or redistribution) | M1 training data | Gitignored, never commit. |
| M1 model checkpoints (`data/m1/`) | Trained by `train/` on CMU-derived data | Own work; CMU allows use in commercial products, but publishing weights is an open question (PLAN.md §9 #9) | M1 model | Local only until the release license is decided. |
| VRoid sample avatars: AvatarSample_D, D_Darkness, E, F, G; Base_Female, Base_Male; HairSample_Female, HairSample_Male; Sakurada_Fumiriya; Sendagaya_Shino | VRoid CC0 sample set (opengameart.org/content/vroid-studio-cc0-models), copies in the owner's `volleyball` repo (`tools/avatars/models/`, unsuffixed files only) | Per embedded VRM meta: CC0 or VRoid Hub license with commercial use, modification and redistribution allowed, no credit required | Reference avatar proportions for retargeting and contact metrics (M3/M4) | Not copied into this repo yet. All VRM 0.x; need conversion for VRM 1.0 work. |
| ~~AvatarSample_M~~ | Same folder | Embedded meta: personal non-profit only, no redistribution, no modification | **Excluded** | Non-commercial terms; don't use without the owner's explicit OK. |
| MHR (Momentum Human Rig) code and assets v1.0.1 (`data/mhr/assets/`) | https://github.com/facebookresearch/MHR (release `assets.zip`) | Apache 2.0 (repo LICENSE and `assets/LICENSE.txt`) | Canonical skeleton (M3 spike onward) | Gitignored for size (4.5 GB unpacked). Apache 2.0 allows redistribution with NOTICE/attribution. |
| pymomentum | https://github.com/facebookresearch/momentum | MIT | MHR runtime dependency (if adopted) | Not installed yet; environment choice pending. |
