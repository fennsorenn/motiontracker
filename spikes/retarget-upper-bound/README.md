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

Re-run 2026-11-11 after the torso/shoulder conversion fix (`mtcore.convert`, rest-to-rest for hips..head
and the shoulders). The first run (2026-11-10) used the old direction-matched torso, which hunched
every avatar forward ~40 deg; see "Corrected" below.

| | copy min gap | copy pair gap / slide | ik min gap | ik pair gap / slide | ik change upper / fore / hand |
|---|---|---|---|---|---|
| palms together | - | 0.6 / 1.8 | - | 0.3 / 0.0 | 0.9 / 2.9 / 2.9 |
| hand to face | -1.7 | 0.5 / 4.1 | -1.0 | 1.9 / 1.7 | 8.0 / 8.8 / 16.8 |
| near miss (6 cm) | 6.7 | 5.1 / 6.1 | 2.8 | 4.3 / 0.1 | 14.5 / 15.8 / 16.0 |

Per avatar: hand to face copy min gap -1.3 to -1.7 cm on the female-base rigs, -2.9 to -3.8 on the
male-base rigs and Sakurada (the performer's own poses: -3.6, see 5.); palms copy pair gap 1.3-1.8 cm
(female base), -1.6 (male).

1. **Plain copy lands contacts roughly right.** Hand to face: about as deep as the performer's own
   (penetrating) pose, but ~4 cm away from the intended spot on the face. Palms: within 2 cm.
2. **Near misses survive copy** (6.7 cm vs the performer's 6.0). IK pulls them to the scaled gap
   (~4.3 cm), which is a choice of the goal, not a need.
3. **Corrections are moderate**: ~9 deg forearm and ~17 deg hand at hand-to-face (up to 28 deg hand on
   some avatars), ~3 deg for palms. Mostly sliding the hand across the face to the right spot.
4. The male-base rigs go deeper under copy (larger heads relative to the arms): ~-3.5 cm.
5. Side finding (unchanged): **the true contact poses penetrate** (performer min gap -1.7 cm palms,
   -3.6 cm hand to face): `mttrain.choreo` aims joint centres (knuckle, fingertip joint) about 1 cm off
   a landmark-derived face point, so the hand's thickness goes into the skin. Harmless for landmark
   training; worth fixing if the choreography drives follow-along videos. It also inflates the copy
   penetration above: with surface-aimed contacts, copy would likely sit near 0 to -1 cm.

**Corrected** (first run, old conversion): copy put the hand ~7 cm inside the head (-9 cm on male
rigs), collapsed near misses to -1 cm, and IK needed 22 deg forearm / 39 deg hand. That was the
hunched torso (head pushed forward and down onto the hand), not avatar proportions.

## Conclusions

- With an upright torso mapping, retargeting at contacts is a refinement, not a rescue: plain copy
  keeps contacts within ~2 cm of the surface and keeps near misses apart, on all 11 avatars.
- What remains is placement: ~4 cm slide on the face, and up to ~3.5 cm penetration on big-head rigs.
  A residual-sized correction (~10-20 deg) covers it.
- Contact intent still matters for a retargeting layer (contact vs near miss decides the goal), but
  the copy no longer destroys the distinction, so the tracker's near-face depth bias is the larger
  problem for contacts by far.

## Caveats

- Upper bound only: true poses and known targets; per-frame IK without temporal smoothness or
  joint limits (hand changes up to ~28 deg not checked for natural wrist ranges).
- Correspondence by bounding-box-normalized part frames is approximate for stylized faces.
- Signed distance from the nearest vertex's normal: reliable for shallow penetration of the face,
  not for hand-hand.
- Three distinct proportion sets (VRoid bases); no chibi or extreme proportions.

Data: `data/spikes/retarget/results.json`, logs (CMU-derived bodies: local only).
