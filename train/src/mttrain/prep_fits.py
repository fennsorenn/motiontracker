"""Batch CMU -> MHR fit for M1 (plan step 2).

Selects CMU clips with little root travel (spike scan, `cmu_scan.json`), fits one performer identity
per subject, then every selected clip's pose at 30 fps with that identity frozen. Writes, under
data/m1/fits/ (gitignored, CMU-derived: never commit):
  <subject>_<clip>.npz   mp (T, P) model parameters, res (T, pairs) residual in cm, fps
  subjects.json          per subject: scale parameters, head calibration offset and head correction (3x3 each)
  index.json             per clip: frames, residuals, glitch metric

    uv run python -m mttrain.prep_fits [--max-travel 60] [--workers 14] [--limit N] [--out DIR]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from mtcore.paths import DATA, REPO_ROOT

OUT = DATA / "m1" / "fits"
SCAN = REPO_ROOT / "spikes" / "m3-mhr-vrm" / "cmu_scan.json"
STEP = 4  # CMU 120 fps -> 30 fps
ARM_PAIRS = ("l_uparm", "r_uparm", "l_lowarm", "r_lowarm", "l_wrist", "r_wrist")
_ch = None


def _worker_init():
    global _ch
    import torch
    torch.set_num_threads(1)  # one process per core; threaded workers oversubscribe the CPU
    from mtcore.mhr import load
    _ch = load()


def _scales(subject, clips):
    from mtcore.cmu_fit import fit_subject_scales, head_offset
    off = head_offset(subject, clips)
    pick = [clips[i] for i in np.linspace(0, len(clips) - 1, min(5, len(clips))).astype(int)]
    scales, post = fit_subject_scales(_ch, subject, pick, head_off=off, head_post=True)
    return subject, {"scales": scales.tolist(), "head_offset": off.tolist(), "head_post": post.tolist()}


def _clip(subject, clip, ident, out):
    from mtcore.cmu import clip_positions
    from mtcore.cmu_fit import PAIRS_M1, fit_clip
    mp, res = fit_clip(_ch, subject, clip, np.asarray(ident["scales"], np.float32), step=STEP,
                       head_off=np.asarray(ident["head_offset"]), head_post=np.asarray(ident["head_post"]))
    names, P = clip_positions(subject, clip, STEP)
    # glitch metric: fastest wrist movement in CMU's own data, m/s
    wr = np.stack([P[:, names.index("lwrist")], P[:, names.index("rwrist")]])
    speed = np.linalg.norm(np.diff(wr, axis=1), axis=-1).max() * (120 / STEP) / 100 if len(P) > 1 else 0.0
    key = f"{subject:02d}_{clip:02d}"
    np.savez_compressed(out / f"{key}.npz", mp=mp, res=res.astype(np.float32), fps=120 / STEP)
    k = [m for m, _, _ in PAIRS_M1]
    arms = res[:, [k.index(a) for a in ARM_PAIRS]]
    return key, {"subject": subject, "frames": len(mp), "arm_res_median": float(np.median(arms)),
                 "arm_res_p95": float(np.percentile(arms, 95)), "all_res_median": float(np.median(res)),
                 "max_wrist_speed": float(speed)}


def main():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")  # inherited by the worker processes
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-travel", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 + 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args()
    out = type(OUT)(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(SCAN.read_text()) if "error" not in r and r["travel_cm"] < a.max_travel]
    clips = sorted((int(r["clip"].split("_")[0]), int(r["clip"].split("_")[1])) for r in rows)
    if a.limit:
        clips = clips[:a.limit]
    by_subject = {}
    for s, c in clips:
        by_subject.setdefault(s, []).append(c)
    print(f"{len(clips)} clips from {len(by_subject)} subjects -> {out}", flush=True)

    t0 = time.time()
    subj_file = out / "subjects.json"
    subjects = json.loads(subj_file.read_text()) if subj_file.exists() else {}
    index_file = out / "index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    with ProcessPoolExecutor(a.workers, initializer=_worker_init) as ex:
        todo = [s for s in by_subject if str(s) not in subjects]
        for f in as_completed([ex.submit(_scales, s, by_subject[s]) for s in todo]):
            s, sc = f.result()
            subjects[str(s)] = sc
        subj_file.write_text(json.dumps(subjects))
        print(f"subject identities: {len(subjects)} in {time.time() - t0:.0f}s", flush=True)

        jobs = [ex.submit(_clip, s, c, subjects[str(s)], out) for s, c in clips
                if f"{s:02d}_{c:02d}" not in index or not (out / f"{s:02d}_{c:02d}.npz").exists()]
        for n, f in enumerate(as_completed(jobs), 1):
            try:
                key, row = f.result()
                index[key] = row
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                print(f"  failed: {e}", flush=True)
            if n % 25 == 0 or n == len(jobs):
                index_file.write_text(json.dumps(index, indent=0))
                frames = sum(r["frames"] for r in index.values())
                print(f"  {n}/{len(jobs)} clips, {frames} frames total, {time.time() - t0:.0f}s", flush=True)
    index_file.write_text(json.dumps(index, indent=0))
    arm = np.array([r["arm_res_median"] for r in index.values()])
    print(f"done: {len(index)} clips, arm residual median of medians {np.median(arm):.2f} cm, "
          f"worst {arm.max():.2f} cm, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
