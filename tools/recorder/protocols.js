// Recording protocols. Each step becomes a labeled segment in session.json; the evaluation
// harness uses the labels (e.g. 'still' for jitter at rest, 'dropout' for hand-loss behaviour).
// Durations are in seconds.

export const PROTOCOLS = {
  'seated-v1': {
    description: 'Seated at a desk, upper body in view. About 70 s.',
    leadIn: 3,
    steps: [
      { label: 'still', duration: 8, prompt: 'Sit naturally, arms resting, look at the camera. Hold still.' },
      { label: 'slow', duration: 10, prompt: 'Slowly raise both arms forward, then lower them again.' },
      { label: 'fast', duration: 10, prompt: 'Quick movements: wave, reach to the sides, point at things.' },
      { label: 'hands', duration: 10, prompt: 'Hands in front of your chest: open, fist, point, count with your fingers.' },
      { label: 'head', duration: 8, prompt: 'Turn your head left and right, nod, tilt it side to side.' },
      { label: 'dropout', duration: 10, prompt: 'Drop your hands out of view and bring them back, a few times.' },
      { label: 'still', duration: 8, prompt: 'Hold still again, arms resting.' },
    ],
  },
  // Like seated-v1, but the two rest segments cover the two rest postures separately (variant
  // tags in session.json). Arms below the frame is a normal streaming posture (PLAN.md §3.3).
  // The contact segments have a known true distance of about zero (criterion 7b, 2026-11-09).
  'seated-v2': {
    description: 'Seated, with separate arms-below-frame and hands-on-desk rest segments and two contact poses. About 83 s.',
    leadIn: 3,
    steps: [
      { label: 'still', variant: 'arms-below-frame', duration: 10, prompt: 'Let your arms hang or rest below the camera view. Look at the camera. Hold still.' },
      { label: 'slow', duration: 10, prompt: 'Slowly raise both arms forward, then lower them again.' },
      { label: 'fast', duration: 10, prompt: 'Quick movements: wave, reach to the sides, point at things.' },
      { label: 'hands', duration: 10, prompt: 'Hands in front of your chest: open, fist, point, count with your fingers.' },
      { label: 'contact', variant: 'palms-together', duration: 6, prompt: 'Press your palms together in front of your chest. Hold.' },
      { label: 'contact', variant: 'hand-to-face', duration: 6, prompt: 'Touch your cheek or chin with one hand. Hold.' },
      { label: 'head', duration: 8, prompt: 'Turn your head left and right, nod, tilt it side to side.' },
      { label: 'dropout', duration: 10, prompt: 'Drop your hands out of view and bring them back, a few times.' },
      { label: 'still', variant: 'hands-on-desk', duration: 10, prompt: 'Rest your hands on the desk where the camera can see them. Hold still.' },
    ],
  },
  // Hands and two-hand interactions for the hand model (dev-notes/plans/m4-hand-model.md, owner
  // 2026-11-12): individual fingers and counting (weak in h2), wrist bends, real palm turns (vs
  // MediaPipe's palm/back flips), contacts between the hands, crossings and one hand in front of the
  // other (MediaPipe's hand swaps and losses). Segments with a known finger state or a true contact are
  // usable as weak labels. "Left" and "right" are your own hands.
  'hands-v1': {
    // Framing: head and shoulders must be in view. Holistic finds the hands from the body (face and pose
    // first, hands next to the pose's wrists); a first take framed on torso and hands only got the face
    // in 0% and the hands in 0-19% of frames (2026-11-12).
    description: 'Fingers, wrists and two-hand interactions. Keep your head and shoulders in view. About 2 min.',
    leadIn: 3,
    steps: [
      { label: 'still', variant: 'hands-on-desk', duration: 6, prompt: 'Head, shoulders and hands in view. Rest your hands on the desk where the camera can see them. Hold still.' },
      { label: 'count', variant: 'left-index-first', duration: 8, prompt: 'Left hand up, palm to the camera. Count 1 to 5: index, middle, ring, little finger, thumb. About one per second.' },
      { label: 'count', variant: 'right-index-first', duration: 8, prompt: 'Now the right hand: count 1 to 5 starting with the index finger.' },
      { label: 'count', variant: 'both-thumb-first', duration: 8, prompt: 'Both hands: count 1 to 5 starting with the thumb, then back down to a fist.' },
      { label: 'fingers', variant: 'single', duration: 10, prompt: 'Both hands open, palms to the camera. Bend each finger down and up on its own: index, middle, ring, little, then thumb.' },
      { label: 'wrist', duration: 8, prompt: 'Keep your forearms still. Bend your wrists up and down, then side to side.' },
      { label: 'turn', duration: 8, prompt: 'Turn your palms toward the camera and away again: slowly first, then quickly.' },
      { label: 'contact', variant: 'palms-together', duration: 5, prompt: 'Press your palms together in front of your chest. Hold.' },
      { label: 'pair', variant: 'fingertips', duration: 8, prompt: 'Touch the fingertips of both hands together, then tap your two index fingers against each other.' },
      { label: 'pair', variant: 'count-on-hand', duration: 10, prompt: 'Left hand open. With your right index finger, touch each finger of your left hand, thumb to little finger.' },
      { label: 'pair', variant: 'rub-clap', duration: 8, prompt: 'Rub your hands together, then clap a few times.' },
      { label: 'pair', variant: 'cross', duration: 8, prompt: 'Cross your hands in front of your chest and uncross them, a few times. Switch which hand is in front.' },
      { label: 'pair', variant: 'occlude', duration: 8, prompt: 'Hold one hand in front of the other, palms to the camera, and slide it slowly across.' },
      { label: 'pair', variant: 'approach', duration: 8, prompt: 'Point both index fingers at each other and bring them slowly together until they almost touch, then apart.' },
      { label: 'dropout', duration: 8, prompt: 'Drop your hands out of view and bring them back, a few times.' },
      { label: 'still', variant: 'hands-on-desk', duration: 6, prompt: 'Rest your hands on the desk again. Hold still.' },
    ],
  },
  free: {
    description: 'No prompts. Press S to start/stop a "still" segment.',
    leadIn: 0,
    steps: [],
  },
};
