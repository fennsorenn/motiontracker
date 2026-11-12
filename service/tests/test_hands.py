"""The finger port against vspark's TypeScript solve (baseline/src/hands-cli.ts) on synthetic hands
(tests/fixtures/make_hands_input.py)."""

import json
import math
from pathlib import Path

from mtservice.hands import finger_bones, solve_hands

FIX = Path(__file__).parent / "fixtures"


def test_matches_typescript_reference():
    frames = [json.loads(l) for l in (FIX / "hands_input.jsonl").read_text().splitlines()]
    refs = [json.loads(l) for l in (FIX / "hands_reference.jsonl").read_text().splitlines()]
    assert len(frames) == len(refs) == 60
    worst = 0.0
    for f, r in zip(frames, refs):
        out = solve_hands({k: f[k] for k in ("leftHand", "rightHand") if k in f})
        assert set(out) == set(r["bones"])
        for b, q in out.items():
            d = min(1.0, abs(sum(a * c for a, c in zip(q, r["bones"][b]))))
            worst = max(worst, 2 * math.degrees(math.acos(d)))
    assert worst < 1e-3


def test_bone_names_and_missing_hands():
    assert len(finger_bones("left")) == 15
    assert set(solve_hands({"leftHand": [[0, 0, 0]] * 21})) == set(finger_bones("left"))
    assert solve_hands({}) == {}
