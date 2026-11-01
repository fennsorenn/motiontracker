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
  free: {
    description: 'No prompts. Press S to start/stop a "still" segment.',
    leadIn: 0,
    steps: [],
  },
};
