"""How well does the face-mesh head pose (`mtcore.facepose`) follow the true head? Rendered sessions (exact ground
truth, real Holistic face mesh) -> calibration of the merger's simulated head observation.

    uv run python -m mttrain.merger.facecal [--sets m1/head1 m1/v1 ...] [--out ../data/merger/calib/face.json]

Per frame with a face: the fit's rotation against the true head rotation in the camera frame. The face model's
frame differs from MHR's head joint frame by a constant rotation (`offset`, estimated over all frames). Measured:
- per axis (turn = about the camera's y, nod = x, tilt = z; Euler YXZ of camera <- face): slope and bias of the
  observed against the true angle, residual std, lag-1 correlation of the residual and its sample-to-sample std (jitter; detector samples at the
  recorder's 10 fps);
- residual against the true turn (wider turns, worse fits) and the share of frames with a face;
- image position: the fit's centre against the projected head joint (bias and noise in frame widths);
- scale against fx / depth (gain and log noise).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from mtcore.facepose import ASPECT, CanonicalFace, fit
from mtcore.mhr import load, skeleton_state

from ..camera import Camera

ROOT = Path(__file__).resolve().parents[4] / "data"
SETS = ["m1/head1", "m1/v1", "m1/rest1", "m1/contact1", "m1/fov1", "m4/pairs3", "m4/hands3", "m4/pairs2", "m4/hands2"]
AXES = ("turn", "nod", "tilt")


def euler(R):
    """camera <- face rotations (..., 3, 3) -> turn, nod, tilt in degrees (Euler YXZ)."""
    return Rot.from_matrix(R.reshape(-1, 3, 3)).as_euler("YXZ", degrees=True).reshape(R.shape[:-2] + (3,))


def collect(seq, ch, canon, head):
    z = np.load(seq / "seq.npz")
    fps = float(z["fps"])
    cam = Camera(z["R"], z["t"], float(z["fx"]))
    st = skeleton_state(ch, z["mp"].astype(np.float64))[:, head]  # (T, 8)
    Rw = Rot.from_quat(st[:, 3:7]).as_matrix()
    pos = st[:, :3] / 100.0
    rows = [json.loads(line) for line in open(seq / "landmarks.jsonl") if line.strip()]
    k = np.array([int(round(r["t"] / 1000 * fps)) for r in rows])
    has = np.array([bool(r.get("face")) and kk < len(st) for r, kk in zip(rows, k)])
    out = {"n": len(rows), "seen": int(has.sum()), "seq": str(seq)}
    if not has.any():
        return out
    F = np.array([r["face"] for r, h in zip(rows, has) if h])
    kk = k[has]
    f = fit(canon, F[..., :468, :])
    C = cam.to_camera(pos[kk])
    u = cam.fx * C[:, 0] / C[:, 2]
    v = cam.fx * C[:, 1] / C[:, 2]  # frame widths, like the fit's centre
    out.update(R_true=np.einsum("ij,tjk->tik", cam.R, Rw[kk]), R_obs=f["R"], scale=f["scale"], centre=f["centre"],
               rms=f["rms"], proj=np.stack([u, v], -1), inv_depth=cam.fx / C[:, 2], frame=kk)
    return out


def mean_rotation(Rs):
    M = Rs.sum(0)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def lag1(x, frame):
    """Lag-1 correlation and step std of x over consecutive detector samples (the smallest frame gap)."""
    gap = np.diff(frame)
    if not len(gap):
        return float("nan"), float("nan")
    ok = gap == gap[gap > 0].min()
    a, b = x[:-1][ok], x[1:][ok]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else float("nan"), float(np.std(b - a)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--out", default=str(ROOT / "merger" / "calib" / "face.json"))
    a = ap.parse_args()
    ch = load()
    head = list(ch.skeleton.joint_names).index("c_head")
    canon = CanonicalFace()
    seqs = []
    for s in a.sets:
        for d in sorted((ROOT / s.split("/")[0] / "renders" / s.split("/")[1]).glob("seq*")):
            if (d / "seq.npz").exists() and (d / "landmarks.jsonl").exists():
                c = collect(d, ch, canon, head)
                c["set"] = s
                seqs.append(c)
    got = [c for c in seqs if c["seen"]]
    offset = mean_rotation(np.concatenate([np.einsum("tji,tjk->tik", c["R_true"], c["R_obs"]) for c in got]))
    report = {"offset": offset.tolist(), "sets": {}}
    for name in a.sets + ["all"]:
        cs = [c for c in got if name == "all" or c["set"] == name]
        if not cs:
            continue
        n = sum(c["n"] for c in seqs if name == "all" or c["set"] == name)
        e_true = np.concatenate([euler(c["R_true"] @ offset) for c in cs])
        e_obs = np.concatenate([euler(c["R_obs"]) for c in cs])
        geo = np.concatenate([np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tij,tkj->tik", c["R_obs"], c["R_true"] @ offset)).as_rotvec(), axis=-1)) for c in cs])
        r = {"frames": n, "seen": float(sum(c["seen"] for c in cs) / n), "geodesic_deg": np.percentile(geo, [50, 90, 99]).round(1).tolist(),
             "true_range_deg": np.percentile(e_true, [5, 95], axis=0).round(1).tolist()}
        for i, ax in enumerate(AXES):
            x, y = e_true[:, i], e_obs[:, i]
            A = np.stack([x, np.ones_like(x)], 1)
            (slope, bias), *_ = np.linalg.lstsq(A, y, rcond=None)
            res_all = y - (slope * x + bias)
            ls, js, means, within = [], [], [], []
            for c in cs:
                et, eo = euler(c["R_true"] @ offset)[:, i], euler(c["R_obs"])[:, i]
                rr = eo - (slope * et + bias)
                means.append(rr.mean())
                within.append(rr - rr.mean())
                l1, j1 = lag1(rr - rr.mean(), c["frame"])
                ls.append(l1); js.append(j1)
            # residual = per-sequence offset (sd_between) + within-sequence AR(1) (sd_within, lag1 at 10 Hz)
            r[ax] = {"slope": round(float(slope), 3), "bias": round(float(bias), 2), "res_std": round(float(res_all.std()), 2),
                     "sd_between": round(float(np.std(means)), 2), "sd_within": round(float(np.concatenate(within).std()), 2),
                     "res_lag1": round(float(np.nanmean(ls)), 3), "jitter": round(float(np.nanmean(js)), 2)}
        turn = np.abs(e_true[:, 0])
        r["geodesic_by_turn"] = {f"{lo}-{hi}": round(float(np.median(geo[(turn >= lo) & (turn < hi)])), 1)
                                 for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 90)) if ((turn >= lo) & (turn < hi)).sum() > 20}
        cen = np.concatenate([c["centre"] - c["proj"] for c in cs])
        lsc = np.concatenate([np.log(c["scale"] / c["inv_depth"]) for c in cs])
        r["centre_bias"] = cen.mean(0).round(4).tolist()
        r["centre_std"] = cen.std(0).round(4).tolist()
        r["scale_gain"] = round(float(np.exp(lsc.mean())), 4)
        r["scale_log_std"] = round(float(lsc.std()), 4)
        r["rms_p50_p95"] = np.percentile(np.concatenate([c["rms"] for c in cs]), [50, 95]).round(4).tolist()
        report["sets"][name] = r
        print(f"{name:12s} seen {r['seen']:.2f}  geo p50/90/99 {r['geodesic_deg']}  " + "  ".join(
            f"{ax} s{r[ax]['slope']:.2f} b{r[ax]['bias']:+.1f} sd{r[ax]['sd_between']:.1f}/{r[ax]['sd_within']:.1f} l{r[ax]['res_lag1']:.2f} j{r[ax]['jitter']:.1f}" for ax in AXES)
            + f"  by turn {r['geodesic_by_turn']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("offset (euler YXZ deg):", euler(offset).round(1).tolist(), "->", a.out)


if __name__ == "__main__":
    main()
