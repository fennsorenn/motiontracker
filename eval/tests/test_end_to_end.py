"""Run the real baseline over a recorder-format session built from vspark's recorded clip."""

import json
import shutil
from pathlib import Path

import pytest

from mteval.report import evaluate, to_markdown
from mteval.session import load_session
from mteval.systems import REPO, run_system

CLIPS = REPO / "baseline" / "test" / "fixtures" / "webcam_arm_clips.json"
pytestmark = pytest.mark.skipif(
    shutil.which("pnpm") is None or not (REPO / "baseline" / "node_modules").exists(),
    reason="baseline not installed (pnpm install in baseline/)",
)


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    frames = json.loads(CLIPS.read_text())["clips"]["hand_dropouts"]["frames"]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n")
    meta = {
        "format": "motiontracker-session",
        "version": 1,
        "name": "hand_dropouts",
        "segments": [{"label": "still", "t0": 0, "t1": 1200}, {"label": "dropout", "t0": 1200, "t1": 2500}],
        "landmarks": {"file": "landmarks.jsonl"},
    }
    (tmp_path / "session.json").write_text(json.dumps(meta))
    return tmp_path


def test_baseline_end_to_end(session_dir: Path):
    session = load_session(session_dir)
    results = [run_system("baseline", session), run_system("baseline-rendered", session)]
    solver, rendered = results
    assert len(solver.track.times) == 25  # one pose per input frame with pose landmarks
    assert 140 <= len(rendered.track.times) <= 150  # ~2.4 s at 60 Hz
    assert "head calibration captured at t=600 ms" in solver.notes

    report = evaluate(session, results)
    assert report["still_source"] == "protocol"
    for name in ("baseline", "baseline-rendered"):
        pb = report["systems"][name]["per_bone"]
        assert pb["presence"]["neck"] == 1.0
        assert pb["rest_jitter"]["leftLowerArm"] is not None
        # The clip's right hand drops in and out; the harness must see those transitions.
        assert pb["dropout_jump"]["rightHand"]["transitions"] > 0
    assert "| arms |" in to_markdown(report)


def test_precomputed_system(session_dir: Path, tmp_path: Path):
    session = load_session(session_dir)
    out = tmp_path / "poses.jsonl"
    out.write_text('{"t": 0, "bones": {"neck": [0, 0, 0, 1]}}\n{"t": 100, "bones": {"neck": [0, 0, 0, 1]}}\n')
    r = run_system(f"model={out}", session)
    assert r.name == "model" and list(r.track.bones) == ["neck"]
