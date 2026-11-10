# Spike: how much could retargeting gain at contacts? (2026-11-10)

**Question.** Before building the retargeting layer (M4, `PLAN.md` §3.10a): with perfect input,
how badly does today's plain rotation copy miss contacts on real avatars, and what would putting
the hand on the avatar's own body cost in arm pose?

## Setup (`run.py`, `avatar_mesh.py`)

```
cd train && ../tools/guarded-run.sh -c 8-15 -- uv run python ../spikes/retarget-upper-bound/run.py [--step 3]
```

- Input: the rendered contact set `contact1` (8 x 20 s, held-out bodies), true MHR poses and
  skinned meshes, holds from `contacts.json`: 14 palms-together, 13 hand-to-face, 7 near misses
  (hand 8-14 cm short of the face point; 6 cm median surface gap). Every 3rd hold frame.
- Avatars: the 11 permitted VRoid samples (`docs/LICENSES-REGISTER.md`; AvatarSample_M excluded).
  They share VRoid base rigs, so they fall into about three proportion sets (female base, male
  base, Sakurada); results cluster accordingly.
- **copy**: MHR -> VRM normalized rotations (`mtcore.convert.MhrToVrm` on the performer's rest
  proportions) + avatar FK. What tracker output gets today.
- **ik** (the upper bound): per hold frame, the copied upper arm, forearm and hand rotations (both
  arms for palms) are adjusted so the avatar's hand point touches the corresponding target point
  (gap 0.3 cm; near misses: the performer's gap x arm-length ratio), with no hand vertex inside the
  face (penalty on planes at the nearest face vertices), staying close to the copy (4 cm^2/rad^2
  arm, 40 for the hand). Ideal input: true pose and known target. No joint limits (VRM has none).
- Correspondence: the performer's closest hand/target vertex pair per frame, mapped through
  normalized part frames (head: eyes + neck; hand: wrist, middle and index/pinky knuckles; scaled
  per axis by the part's rest bounding box) and snapped to the avatar's nearest surface vertex.
- Measures on the meshes: **min gap** = closest signed hand-to-face distance (sign from the face
  normal at the nearest vertex; negative = inside the head; not used for palms, where two
  non-convex hands make it unreliable); **pair gap / slide** = offset of the corresponding points
  along / across the target normal; **change** = IK vs copy, upper arm and forearm direction, hand
  rotation.

## Results (medians over events; cm, deg)

| | copy min gap | copy pair gap / slide | ik min gap | ik pair gap / slide | ik change upper / fore / hand |
|---|---|---|---|---|---|
| palms together | - | 3.1 / 3.2 | - | 0.3 / 0.0 | 1.5 / 5.3 / 5.4 |
| hand to face | **-7.3** | -7.1 / 4.8 | -1.0 | 1.8 / 1.2 | 11.1 / 22.4 / 38.8 |
| near miss (6 cm) | -1.1 | -2.1 / 4.9 | 2.9 | 4.6 / 0.0 | 8.6 / 14.9 / 16.3 |

Per avatar: hand to face copy min gap -5.8 to -7.4 cm on the female-base rigs, -9.1 to -9.4 on the
male-base rigs and Sakurada; palms copy pair gap 3.3-3.9 cm (female base), 1.2-1.3 (male).

1. **Hand to face is the big one.** Plain copy puts the hand about 7 cm *inside* the avatar's head
   (9 cm on the male rigs), touching about 5 cm away from the intended spot. The VRoid heads are
   large relative to the arms, so the performer's arm angles end inside them.
2. **Fixing it is a real pose change, not a nudge.** Putting the hand on the face surface takes
   ~22 deg at the forearm and ~39 deg at the hand (even with hand turns 10x as expensive as arm
   turns): the fingers have to come out of the bigger head. A residual -1 cm remains (fingers).
3. **Near misses collapse.** A 6 cm near miss becomes a touch or slight penetration (-1.1 cm) under
   copy; IK keeps ~4.6 cm (the scaled gap). Retargeting has to know contact from near miss.
4. **Palms survive copy reasonably** (3 cm apart, 3 cm sliding): both arms scale alike. A ~5 deg
   correction closes them.
5. Side finding: **the true contact poses penetrate** (performer min gap -1.7 cm palms, -3.6 cm
   hand to face): `mttrain.choreo` aims joint centres (knuckle, fingertip joint) about 1 cm off a
   landmark-derived face point, so the hand's thickness goes into the skin. Harmless for landmark
   training; worth fixing if the choreography drives follow-along videos.

## Conclusions

- On these avatars, retargeting is worth a lot at hand-to-face contacts: ~7 cm penetration under
  plain copy is visible on every avatar, independent of tracker quality.
- It needs the intent: the same ~6 cm difference separates a near miss from a contact, so the
  layer must be told which one is meant. Today the tracker can't (MediaPipe puts touching hands
  8-26 cm in front of the face). Contact intent has to come from the lifter first.
- Palms together is a small correction; hand to face is a large one (forearm/hand), so the layer
  must be allowed to change the pose substantially near the face, i.e. it can't be a small residual.

## Caveats

- Upper bound only: true poses and known targets; per-frame IK without temporal smoothness or
  joint limits (39 deg hand changes not checked for natural wrist ranges).
- Correspondence by bounding-box-normalized part frames is approximate for stylized faces.
- Signed distance from the nearest vertex's normal: reliable for shallow penetration of the face,
  not for hand-hand.
- Three distinct proportion sets (VRoid bases); no chibi or extreme proportions.

Data: `data/spikes/retarget/results.json`, logs (CMU-derived bodies: local only).
