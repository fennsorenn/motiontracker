# avatar-video: side-by-side comparison videos

Renders a recorded session as one video: the mirrored webcam next to one VRM avatar per tracking
system, each posed from that system's track. Every avatar panel has a side-view inset, because the
front view hides reach toward the camera (where the systems differ most). The protocol segment and
its prompt run along the bottom.

```
# 1. tracks (harness system syntax, so each panel is exactly what mteval measures)
uv run --project eval python tools/avatar-video/prepare.py sessions/<session> data/videos/<name> \
    --system baseline-rendered --system m1-v8-glide=data/m1/runs/m1-v8/session.poses.jsonl
# 2. frames + video -> data/videos/<name>/compare.mp4  (optionally --from s --to s, --size px)
tools/guarded-run.sh -c 8-15 -- node tools/avatar-video/render.mjs data/videos/<name> <avatar.vrm> sessions/<session>
```

- Posing follows vspark's viewport: `resetNormalizedPose` + `setNormalizedPose` with the normalized
  bone rotations, three.js 0.168.0 and @pixiv/three-vrm 3.5.2 (vspark's versions, from jsDelivr).
  VRM 0.x avatars are turned to face the camera (`VRMUtils.rotateVRM0`) and get the rotations with x
  and z negated, as three-vrm does for VRM 0.x animation. Spring bones run at 1/60 s per frame.
- Bones: limb bones only where every system outputs them (the learned models have no fingers, so
  the baseline's fingers stay at rest too); the torso/neck/head chain is kept per system, since
  systems split the same head pose differently (the baseline has no hips, upperChest or head bone).
  `prepare.py --all-bones` keeps everything.
- Not reproduced: vspark's arm calibration, tracking-loss fades, root motion and IK mode. No
  retargeting: rotations are copied onto the avatar as they are.
- Webcam panel: the session's `video.webm` mirrored, resampled to 60 fps; landmark times count from
  the recorder start, so expect about one frame of offset.
- Renders in headless Chrome with software WebGL (~7 frames/s at 2000 x 600 on 8 cores: ~10 min for a 67 s session). Outputs contain the
  owner's webcam video: keep them in gitignored `data/videos/`, never commit them.
