# Spike: where MediaPipe's pose model represents hand-vs-face depth (2026-11-09)

**Question.** On rendered contact sessions, MediaPipe puts a hand that touches the face 8-26 cm in
front of the nose, about the same as for near misses that really are 5-19 cm in front (M1 plan, step
7e). Is the true depth still present somewhere inside the pose landmark model? If it is linearly
readable from the features the output heads read, a closed-form edit of the heads (no gradient
training, standard MediaPipe runtime) could fix it. Otherwise a fine-tune would have to reach
deeper, or the cue isn't in the image at the model's resolution.

## Setup

- Model: `pose_landmarks_detector.tflite` from the Holistic bundle the recorder uses (float16 v1),
  run with LiteRT, delegates off and all intermediate tensors kept (68 activation layers).
- **Both landmark heads are linear** in one shared 768-d feature (`re_lu_55`, 2x2x192): image
  landmarks (39 x 5) and world landmarks (39 x 3) are each a single 2x2 conv over it.
- Crop: MediaPipe's tracking ROI from its own auxiliary landmarks 33/34 (side 2.5 x their distance,
  rotated upright), refined from the model's own output for 4 rounds; pose flag ~0.99.
- Data (`make_configs.py`, then `tools/render/render_blender.py`): 80 configurations (held-out
  bodies, seated, random desk camera with 53-129 deg FOV, face point chin/cheek/forehead/mouth,
  fingertip or palm). Each has the hand at 0.5, 4, 8 and 14 cm from the face along the face normal,
  everything else fixed: 320 frames. MHR IK as in `mttrain.choreo`.
- Target: true depth of the active hand (index/pinky knuckle midpoint) minus the nose, camera frame.
  It goes from -0.6 cm at contact to -8.9 cm at 14 cm (the face normal isn't along the view axis).
- Probes (`analyze.py`): ridge regression from each layer's activations at the hand and the nose
  plus the spatial mean, 5-fold with whole configurations held out, alpha by inner CV. "Within" =
  R2 after removing each configuration's mean, i.e. does it follow the gap.

```
cd train && uv run python ../spikes/mp-depth-probe/make_configs.py --n 80
for d in data/spikes/mp-depth/configs/c*; do blender -b -t 8 --python tools/render/render_blender.py -- $d/seq.npz $d/frames; done
cd spikes/mp-depth-probe && uv run python probe.py run && uv run python analyze.py
```

## Results

**MediaPipe's own output moves the wrong way.** As the hand moves from contact to 14 cm in front,
the true hand-minus-nose depth goes -0.6 -> -8.9 cm; MediaPipe's world output goes -11.1 -> -8.2 cm
(image z the same way). Within configurations its slope against the truth is -0.26. At contact it
is 10.5 cm off; at 14 cm, where the hand really is in front, it happens to be roughly right.

**The depth isn't linearly readable where the heads read.**

| Features | R2 overall | R2 within |
|---|---|---|
| Head features `re_lu_55` (768) | 0.00 | -0.02 |
| Early layers, 128x128 and 64x64 | <= 0.22 | <= 0 |
| Mid layers, 32x32 to 8x8 (best: `re_lu_10..17`, `re_lu_23`) | 0.2-0.29 | 0.15-0.22 |
| Late layers, 4x4 and 2x2 | ~0 | ~0 |
| Control: head features, shuffled target | -0.01 | |

At best a weak trace exists in the middle of the network, at the 16x16 and 8x8 stages. It is gone
by the 4x4 stage. The contrast changes from contact to 14 cm point in a slightly shared direction
only in the same mid layers (mean cosine 0.11-0.18; ~0 elsewhere).

## Conclusions

1. **A closed-form edit of the output heads can't fix hand-at-face depth.** The 768 features the
   heads read carry no linear trace of it on these renders.
2. **A fine-tune would have to reach the mid network** (16x16 and below), where the trace is weak.
   That is a full fine-tune in practice, with the render domain-gap risk discussed for it.
3. **The cue is probably near the model's resolution limit.** In the 256 px full-body crop, a hand
   moving 8 cm toward the camera at ~70-100 cm distance grows by ~10%, a few pixels. The Holistic
   hand landmarker works on its own high-resolution hand crop, so hand size relative to face size is
   the more promising depth cue, fed to the lifter (which already sees history, so it can use the
   scale change over time too).

## Follow-up: hand size as a depth cue (`hand.py`)

MediaPipe's hand landmark model (Holistic bundle) on its own 224 px hand crop, tracked from the true
pose hand and then from its own landmarks. Its world landmarks are metric and its image landmarks are
pixels, so the hand's scale gives its distance (z ~ fx / pixels-per-metre, weak perspective).

- **Detection:** the hand was found (presence > 0.5) in 40% of frames, 39% at contact; only 12 of
  80 configurations at all four gaps. Bare render hands against the face are hard for it; the real
  session had hands in about half of all frames.
- **When found, the distance is unbiased:** median |error| 5.0 cm (7.7%) at ~60 cm, median signed
  +0.4 cm. Compare the pose model: 10.5 cm off at contact, and systematic.
- **Within configurations it follows the gap the right way, but noisily:** slope +1.4, correlation
  +0.38, R2 0.15, within error 3.4 cm, against a true depth range of 8 cm per configuration.

So the hand's apparent size is a real, unbiased but noisy depth cue, unlike the pose model's
systematic error. Noise averages out over the lifter's history; a bias wouldn't. The face's distance
(for hand-minus-face depth) would come the same way from the face landmarks with a per-person face
size; not tested here. The detection rate is the main risk.

### Orientation and the elbow (owner's questions)

- **Orientation matters.** The simple size estimate (3D spread of the model's world landmarks vs
  2D spread) is unbiased on average but edge-on hands read too far: median |error| 9.3 cm with the
  palm edge-on, 3.5-4.2 cm when it faces the camera. Fitting the hand's orientation (2x3 projection
  from world to image landmarks, scale from its singular values) lowers the within-configuration
  error to 2.7 cm but reads 8-12 cm too close: the hand model's own 3D hand shape is not accurate
  enough for a hand-made estimator. Better to give the lifter the 21 landmarks and let it learn.
- **The elbow is a strong geometric cue.** From contact to 14 cm the forearm turns toward the camera
  (out-of-plane angle +20 -> +40 deg, median) and its 2D length shrinks by 24%, more than the hand's
  apparent size changes. The lifter already gets elbow and wrist image positions. But MediaPipe's 2D
  keeps only part of it: 15% shrink instead of 24%, within-configuration correlation 0.38 (elbow 13
  px, wrist 16 px off on an 83 px forearm). It also needs the forearm length (per person) and can't
  tell toward from away by itself.
- So near the face all three cues are degraded in MediaPipe in ways our synthesis doesn't model:
  world depth biased toward the camera, 2D foreshortening damped, hand landmarks often missing.

## Caveats

- Renders only: plain MHR body, EEVEE lighting, no clothes or hair; real images may carry cues the
  renders lack (contact shadows, skin deformation). Re-run the probes on real frames once there is
  two-camera depth.
- Single-cell features at the hand position; 320 samples. Weak linear signals may be missed, but a
  usable signal for a head edit would have to show up here.

Data (renders, features, logs): `data/spikes/mp-depth/` (CMU-derived bodies: local only).
