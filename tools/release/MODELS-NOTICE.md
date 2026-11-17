# motiontracker model bundles

The ONNX models and tables in this archive are the live runtime's inputs (`mtservice`):

- `data/merger/runs/mg3/live/`: merger `mg3` with its body model `m1-v14` and pair hand model `p3`
  (`body.onnx`, `hands.onnx`, `merger.onnx`, `merger.json`) and the retargeting mesh `retarget.npz`;
- `data/m1/runs/m1-v10/live/`: body model `m1-v10` alone (`--no-merger`).

Training data: the body models and the merger were trained on motion derived from the CMU Graphics Lab Motion
Capture Database. The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with
funding from NSF EIA-0196217. No CMU data is contained in this archive.

`retarget.npz` holds vertex positions and body-part labels of MHR's mean body mesh. The `.json` tables hold rest-pose
data of MHR (Momentum Human Rig, Apache 2.0,
https://github.com/facebookresearch/MHR), MediaPipe's canonical face Procrustes points (Apache 2.0) and the rest bone
positions of VRoid's CC0 sample avatar AvatarSample_D.

Unpack this archive into the root of the matching motiontracker source release (same version).
