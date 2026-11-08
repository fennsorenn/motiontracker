"""mteval command line.

  uv run mteval evaluate SESSION [--system baseline] [--system baseline-rendered]
                         [--system name=poses.jsonl] [--system name-rendered=poses.jsonl]
                         [--system name-glide=poses.jsonl] [--out DIR] [--auto-still] [--no-calibration]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import evaluate, write_report
from .session import load_session
from .systems import run_system

REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mteval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="run systems over a session and write a report")
    ev.add_argument("session", help="session directory (from the recorder) or landmarks .jsonl")
    ev.add_argument("--system", action="append", help="baseline, baseline-rendered, name=poses.jsonl, name-rendered=poses.jsonl (through the render filter), or name-glide=poses.jsonl (60 Hz upsampling, no smoothing)")
    ev.add_argument("--out", help="report directory (default: <repo>/reports/<session folder>)")
    ev.add_argument("--auto-still", action="store_true", help="detect still segments if the session has none")
    ev.add_argument("--no-calibration", action="store_true", help="skip the baseline head calibration")
    args = ap.parse_args(argv)

    session = load_session(args.session)
    systems = args.system or ["baseline", "baseline-rendered"]
    results = [run_system(s, session, calibrate=not args.no_calibration) for s in systems]
    report = evaluate(session, results, auto_still=args.auto_still)
    out = Path(args.out) if args.out else REPO_ROOT / "reports" / session.name
    j, m = write_report(report, out)
    print(f"wrote {m} and {j}")


if __name__ == "__main__":
    main()
