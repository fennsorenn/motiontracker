"""Evaluate systems on a session and write a JSON + markdown report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import metrics as M
from .session import Session, auto_still_segments
from .systems import SystemResult


def evaluate(session: Session, results: list[SystemResult], auto_still: bool = False, gt=None) -> dict:
    """`gt`: optional ground-truth PoseTrack (rendered synthetic sessions) for the labeled metrics."""
    segments = list(session.segments)
    still = session.segments_labeled("still")
    still_source = "protocol"
    if not still and auto_still:
        still = auto_still_segments(session)
        segments += still
        still_source = "auto-detected"
    elif not still:
        still_source = "none"

    report = {
        "session": session.name,
        "frames": len(session.frames),
        "duration_s": float((session.times[-1] - session.times[0]) / 1000) if session.frames else 0.0,
        "segments": [{"label": s.label, "variant": s.variant, "t0": s.t0, "t1": s.t1} for s in segments],
        "still_source": still_source,
        "systems": {},
    }
    for r in results:
        tr = M.with_head_global(r.track)
        report["systems"][r.name] = {
            "notes": r.notes,
            "output_frames": len(tr.times),
            "per_bone": {
                "speed": M.speed_percentiles(tr, segments),
                "rest_jitter": M.rest_jitter(tr, still),
                "rest_noise": M.rest_noise(tr, still),
                "twitch_rate": M.twitch_rate(tr, still),
                "presence": M.presence(tr),
                "dropout_jump": M.dropout_jump(tr, session),
                "lag_vs_detector": M.lag_vs_detector(tr, session),
            },
            "hand_position": M.hand_position(tr, session),
            "contact": M.contact(tr, session),
            "ground_truth": M.ground_truth_metrics(r.track, gt) if gt is not None else None,
            "per_still_segment": [
                {
                    "segment": still_name(s),
                    "rest_noise": M.rest_noise(tr, [s]),
                    "twitch_rate": M.twitch_rate(tr, [s]),
                }
                for s in still
            ],
        }
    report["summary"] = summarize(report)
    report["still_segment_summary"] = summarize_still_segments(report)
    return report


def still_name(s) -> str:
    span = f"{s.t0 / 1000:.0f}-{s.t1 / 1000:.0f} s"
    return f"{s.variant} ({span})" if s.variant else f"still {span}"


def _median(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def summarize(report: dict) -> dict:
    """Median over the bones of each group, per system and metric."""
    out: dict[str, dict] = {}
    for name, sys in report["systems"].items():
        pb = sys["per_bone"]
        groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for bone in pb["presence"]:
            g = groups[M.bone_group(bone)]
            g["rest_jitter"].append(pb["rest_jitter"].get(bone))
            g["rest_noise"].append(pb["rest_noise"].get(bone))
            g["twitch_rate"].append(pb["twitch_rate"].get(bone))
            g["presence"].append(pb["presence"].get(bone))
            sp = pb["speed"].get(bone, {}).get("all")
            g["speed_p99"].append(sp["p99"] if sp else None)
            dj = pb["dropout_jump"].get(bone)
            g["dropout_jump"].append(dj["mean_peak"] if dj else None)
            lag = pb["lag_vs_detector"].get(bone)
            g["lag_ms"].append(lag["lag_ms"] if lag and (lag["corr"] or 0) >= 0.3 else None)
        out[name] = {grp: {k: _median(v) for k, v in m.items()} for grp, m in groups.items()}
    return out


def summarize_still_segments(report: dict) -> dict:
    """Per system and still segment: median over the bones of each group."""
    out: dict[str, dict] = {}
    for name, sys in report["systems"].items():
        out[name] = {}
        for seg in sys.get("per_still_segment", []):
            groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for key in ("rest_noise", "twitch_rate"):
                for bone, v in seg[key].items():
                    groups[M.bone_group(bone)][key].append(v)
            out[name][seg["segment"]] = {g: {k: _median(v) for k, v in m.items()} for g, m in groups.items()}
    return out


METRIC_LABELS = [
    ("rest_jitter", "Rest jitter (deg/s)"),
    ("rest_noise", "Rest noise (deg)"),
    ("twitch_rate", "Twitches/min at rest"),
    ("speed_p99", "Speed p99 (deg/s)"),
    ("dropout_jump", "Hand dropout peak (deg/s)"),
    ("lag_ms", "Lag vs detector (ms)"),
    ("presence", "Presence"),
]
GROUP_ORDER = ["head_global", "torso_head", "shoulders", "arms", "hands", "fingers"]


def to_markdown(report: dict) -> str:
    lines = [
        f"# Evaluation: {report['session']}",
        "",
        f"{report['frames']} input frames, {report['duration_s']:.1f} s. "
        f"Still segments: {report['still_source']}.",
        "",
        "Values are the median over the bones in each group. Lag is only counted where the "
        "correlation with the detector signal is at least 0.3. head_global is the head's orientation "
        "relative to the avatar root (product of the torso, neck and head bones); compare torso and "
        "head across systems with it, since systems split the chain differently. See eval/README.md "
        "for definitions.",
        "",
    ]
    for name, sys in report["systems"].items():
        lines.append(f"- **{name}**: {sys['output_frames']} output frames; {'; '.join(sys['notes'])}")
    lines.append("")
    systems = list(report["summary"])
    for key, label in METRIC_LABELS:
        lines += [f"## {label}", "", "| Group | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        for grp in GROUP_ORDER:
            cells = []
            for s in systems:
                v = report["summary"][s].get(grp, {}).get(key)
                cells.append("–" if v is None else f"{v:.2f}" if key == "presence" else f"{v:.1f}")
            if any(c != "–" for c in cells):
                lines.append(f"| {grp} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Hand position on the performer's skeleton (criterion 7)", "",
              "Mean of the per-side medians. 2D: wrist error in the camera plane against the image "
              "landmarks (cm, main score). Forearm direction: angle in the image (deg, guard against errors "
              "along the viewing axis). 3D: against the world landmarks (cm; MediaPipe's depth is unreliable). "
              "In view = shoulder, elbow and wrist visibility at least 0.8.", "",
              "| Measure | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for key, label in (("in_view_2d", "2D, in view (cm)"), ("all_2d", "2D, all (cm)"),
                       ("forearm_dir_2d", "forearm direction, in view (deg)"),
                       ("in_view", "3D, in view (cm)"), ("all", "3D, all (cm)")):
        cells = []
        for s in systems:
            hp = report["systems"][s].get("hand_position") or {}
            meds = [hp[side][key]["median"] for side in ("left", "right") if hp.get(side) and hp[side].get(key)]
            cells.append(f"{np.mean(meds):.1f}" if meds else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    if any(report["systems"][s].get("ground_truth") for s in systems):
        lines += ["## Against ground truth", "",
                  "Global orientation error (deg, median over frames, mean over the bones of a group; global = "
                  "relative to the avatar root, so chain splits don't matter), wrist position error relative to "
                  "the shoulder on a common arm (cm), onset lag (ms) and peak-velocity ratio (arms).", "",
                  "| Measure | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
        def gtcell(s, fn):
            g = report["systems"][s].get("ground_truth")
            v = fn(g) if g else None
            return "–" if v is None else f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
        def grp(g, key, bones, field):
            vals = [g[key][b][field] for b in bones if g[key].get(b) and g[key][b].get(field) is not None]
            return float(np.mean(vals)) if vals else None
        arm_up = [f"global:{s}UpperArm" for s in ("left", "right")]
        arm_lo = [f"global:{s}LowerArm" for s in ("left", "right")]
        hands = [f"global:{s}Hand" for s in ("left", "right")]
        rows = [("head (deg)", lambda g: grp(g, "global_rotation_error", ["global:head"], "median")),
                ("upper arms (deg)", lambda g: grp(g, "global_rotation_error", arm_up, "median")),
                ("forearms (deg)", lambda g: grp(g, "global_rotation_error", arm_lo, "median")),
                ("hands (deg)", lambda g: grp(g, "global_rotation_error", hands, "median")),
                ("wrist position (cm)", lambda g: float(np.mean([g["wrist_error"][x]["median"] for x in ("left", "right") if g["wrist_error"].get(x)])) if any(g["wrist_error"].values()) else None),
                ("wrist position p90 (cm)", lambda g: float(np.mean([g["wrist_error"][x]["p90"] for x in ("left", "right") if g["wrist_error"].get(x)])) if any(g["wrist_error"].values()) else None),
                ("forearm onset lag (ms)", lambda g: grp(g, "onset_lag", arm_lo, "lag_ms")),
                ("forearm peak-velocity ratio", lambda g: grp(g, "peak_velocity_ratio", arm_lo, "median"))]
        for label, fn in rows:
            lines.append(f"| {label} | " + " | ".join(gtcell(s, fn) for s in systems) + " |")
        lines.append("")
    variants = sorted({v for s in systems for v in (report["systems"][s].get("contact") or {})})
    if variants:
        lines += ["## Contact (criterion 7b, cm)", "",
                  "Median distance during protocol contact segments, true distance about 0 (wrists for palms "
                  "together, wrist to nose for hand to face). MediaPipe: the same from its own landmarks.", "",
                  "| Contact | MediaPipe | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
        for v in variants:
            c = {s: (report["systems"][s].get("contact") or {}).get(v) for s in systems}
            mp = next((x["mediapipe_cm"] for x in c.values() if x), None)
            lines.append(f"| {v} | {'–' if mp is None else f'{mp:.1f}'} | "
                         + " | ".join("–" if not c[s] else f"{c[s]['system_cm']:.1f}" for s in systems) + " |")
        lines.append("")
    seg_summary = report.get("still_segment_summary", {})
    seg_names = list(next(iter(seg_summary.values()), {}))
    if len(seg_names) > 1:
        lines += ["## Rest by still segment", "",
                  "Rest noise (deg) / twitches per minute, per still segment. The tables above pool them.", ""]
        for s in systems:
            lines += [f"**{s}**", "", "| Group | " + " | ".join(seg_names) + " |", "|---|" + "---|" * len(seg_names)]
            for grp in GROUP_ORDER:
                cells = []
                for seg in seg_names:
                    g = seg_summary[s].get(seg, {}).get(grp, {})
                    n, tw = g.get("rest_noise"), g.get("twitch_rate")
                    cells.append("–" if n is None and tw is None else
                                 f"{'–' if n is None else f'{n:.1f}'} / {'–' if tw is None else f'{tw:.1f}'}")
                if any(c != "–" for c in cells):
                    lines.append(f"| {grp} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "report.json"
    m = out_dir / "report.md"
    j.write_text(json.dumps(report, indent=1))
    m.write_text(to_markdown(report))
    return j, m
