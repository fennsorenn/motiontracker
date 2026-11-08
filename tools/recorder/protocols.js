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
  free: {
    description: 'No prompts. Press S to start/stop a "still" segment.',
    leadIn: 0,
    steps: [],
  },
};
