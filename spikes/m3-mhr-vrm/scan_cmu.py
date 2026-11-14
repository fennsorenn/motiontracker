"""Rank CMU clips by upper-body activity: little root travel, much hand motion relative to the chest."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mtcore.cmu import CMU_ROOT, Skeleton, read_amc


def score(path):
    try:
        sid = path.parent.name
        sk = Skeleton(path.parent / f"{sid}.asf")
        frames = read_amc(path)
        frames = frames[:: max(1, len(frames) // 60)][:60]
        P = [sk.fk(f) for f in frames]
        root = np.array([p["root"] for p in P])
        chest = np.array([p["thorax"] for p in P])
        travel = float(np.linalg.norm(np.ptp(root[:, [0, 2]], axis=0)))
        hands = []
        near_head = 0.0
        for side in "lr":
            w = np.array([p[f"{side}wrist"] for p in P]) - chest
            hands.append(float(np.mean(np.linalg.norm(np.diff(w, axis=0), axis=1))))
            head = np.array([p["head"] for p in P])
            near_head = max(near_head, float(np.mean(np.linalg.norm(np.array([p[f"{side}wrist"] for p in P]) - head, axis=1) < 25)))
        return {"clip": path.stem, "frames": len(read_amc(path)) if False else None, "travel_cm": travel,
                "hand_motion": float(np.mean(hands)), "near_head": near_head}
    except Exception as e:  # noqa: BLE001
        return {"clip": path.stem, "error": str(e)}


if __name__ == "__main__":
    paths = sorted(CMU_ROOT.glob("*/*.amc"))
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(score, paths, chunksize=8))
    ok = [r for r in rows if "error" not in r]
    Path("cmu_scan.json").write_text(json.dumps(rows, indent=0))
    still = [r for r in ok if r["travel_cm"] < 60]
    print(f"{len(rows)} clips, {len(rows) - len(ok)} errors, {len(still)} with < 60 cm root travel")
    print("top upper-body clips (hand motion relative to chest, little travel):")
    for r in sorted(still, key=lambda r: -r["hand_motion"])[:15]:
        print(f"  {r['clip']:8s} travel {r['travel_cm']:5.1f} cm  hand motion {r['hand_motion']:5.2f}  hands near head {r['near_head']:.0%}")
    print("most hand-near-head clips:")
    for r in sorted(still, key=lambda r: -r["near_head"])[:8]:
        print(f"  {r['clip']:8s} travel {r['travel_cm']:5.1f} cm  hand motion {r['hand_motion']:5.2f}  hands near head {r['near_head']:.0%}")
