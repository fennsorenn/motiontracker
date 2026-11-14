"""Synthetic hand world landmarks for tests/test_hands.py (no recorded data): a made-up 21-point hand,
randomly curled, splayed, rotated and jittered, both sides. Regenerate the reference with:

    uv run python tests/fixtures/make_hands_input.py > tests/fixtures/hands_input.jsonl
    (cd ../baseline && npx tsx src/hands-cli.ts) < tests/fixtures/hands_input.jsonl > tests/fixtures/hands_reference.jsonl
"""
import json

import numpy as np

# A flat right-ish hand in metres, wrist at the origin, fingers along -y (image up), palm toward -z.
BASE = {0: (0, 0, 0), 1: (-0.025, -0.02, 0), 5: (-0.022, -0.085, 0), 9: (0, -0.09, 0), 13: (0.02, -0.085, 0),
        17: (0.038, -0.075, 0)}
SEG = {1: [0.035, 0.03, 0.025], 5: [0.04, 0.025, 0.02], 9: [0.045, 0.028, 0.022], 13: [0.04, 0.026, 0.02],
       17: [0.03, 0.02, 0.018]}


def rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def hand(rng, mirror):
    pts = np.zeros((21, 3))
    for root, lens in SEG.items():
        p = np.array(BASE[root], float)
        pts[root] = p
        d = p / np.linalg.norm(p) if root != 1 else np.array([-0.7, -0.7, 0.0])
        d = rot([0, 0, 1], rng.uniform(-0.25, 0.25)) @ d  # splay
        for k, L in enumerate(lens):
            d = rot([d[1], -d[0], 0], rng.uniform(-0.3, 1.6)) @ d  # curl toward the palm (or back)
            p = p + L * d
            pts[root + 1 + k] = p
    if mirror:
        pts[:, 0] *= -1
    R = rot(rng.normal(size=3), rng.uniform(0, np.pi))
    return pts @ R.T + rng.normal(scale=0.002, size=pts.shape)


rng = np.random.default_rng(7)
for i in range(60):
    f = {"t": i * 100.0}
    if i % 3 != 1:
        f["rightHand"] = hand(rng, False).round(6).tolist()
    if i % 3 != 2:
        f["leftHand"] = hand(rng, True).round(6).tolist()
    print(json.dumps(f))
