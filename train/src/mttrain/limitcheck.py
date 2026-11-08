"""Exact MHR joint-limit check for M1 output (success criterion 8: near zero in all conditions).

    uv run python -m mttrain.limitcheck <ckpt> sessions/<session> [--gt-samples 64]

The model predicts joint orientations, not MHR parameters, and trains against a cheap proxy box
(`losses.limit_box`). Here the predicted orientations of the 17 output joints are fitted with MHR's
own parameters by IK (pose parameters only, template proportions, limits switched off), and the
fitted parameters are checked against MHR's [Limits]. Reports frames with any parameter past its
limit by more than 1 and 6 degrees, the worst parameters, and the IK orientation residual (a large
residual means the pose isn't reachable by MHR's joints at all, which is itself implausible).

Conditions: the session as recorded, plus stressed copies of it: tracking loss (detector returns
nothing for 2 s windows), low confidence (all visibilities x0.3), extra noise, arms out of frame
(hands pushed below the frame with near-zero visibility and drifting guesses), and glitches
(random landmark jumps). `--gt-samples` also runs the check on synthetic ground-truth targets,
which are clamped to the limits, to validate the checker itself (expect ~0).
"""

import argparse

import numpy as np
import pymomentum.geometry as geom
import pymomentum.solver as S
import torch

from mtcore.mhr import FLEX_T, limits

from .infer import Runner, mat_to_quat, read_session
from .skeleton import OUT_JOINTS

EXCESS = (np.radians(1.0), np.radians(6.0))


class LimitChecker:
    def __init__(self, runner):
        self.r = runner
        ch = runner.ch
        self.pn = list(ch.parameter_transform.names)
        lim = limits()
        self.check = [(i, n, *lim[n]) for i, n in enumerate(self.pn)
                      if n in lim and lim[n][1] > lim[n][0] and not n.startswith("scale_")]
        self.active = torch.tensor([not (n.startswith("scale_") or n in FLEX_T or n in ("root_tx", "root_ty", "root_tz"))
                                    for n in self.pn])
        self.idx = [runner.jn.index(j) for j in OUT_JOINTS]

    def fit(self, G, chunk=500):
        """MHR world joint rotations (T, J, 3, 3) -> fitted parameters (T, P), orientation residual (T, 17) deg."""
        ch = self.r.ch
        opts = S.SolverOptions()
        opts.max_iter = 60
        tgt_all = mat_to_quat(G[:, self.idx])
        mps, res = [], []
        for s in range(0, len(G), chunk):
            tgt = torch.tensor(tgt_all[s:s + chunk], dtype=torch.float32)
            n = len(tgt)
            cons = dict(orientation_cons_parents=torch.tensor(self.idx, dtype=torch.int32),
                        orientation_cons_offsets=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * len(self.idx)),
                        orientation_cons_weights=torch.ones(len(self.idx)),
                        orientation_cons_targets=tgt)
            mp = S.solve_ik(ch, self.active, torch.zeros(n, len(self.pn)),
                            [S.ErrorFunctionType.Orientation, S.ErrorFunctionType.Limit], torch.tensor([1.0, 0.0]),
                            opts, **cons).detach().numpy()
            st = np.asarray(geom.model_parameters_to_skeleton_state(ch, mp.astype(np.float32)))
            q = st[:, self.idx, 3:7]
            dot = np.abs((q * tgt_all[s:s + chunk]).sum(-1)) / np.linalg.norm(q, axis=-1)
            res.append(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))
            mps.append(mp)
        return np.concatenate(mps), np.concatenate(res)

    @staticmethod
    def _excess(x, lo, hi):
        """Distance past [lo, hi]; angles are taken modulo 2 pi around the range's centre."""
        mid = 0.5 * (lo + hi)
        x = (x - mid + np.pi) % (2 * np.pi) - np.pi + mid
        return np.maximum(0, lo - x) + np.maximum(0, x - hi)

    def report(self, mp, res):
        ex = np.stack([self._excess(mp[:, i], lo, hi) for i, _, lo, hi in self.check], 1)
        worst = ex.max(1)
        names = [n for _, n, _, _ in self.check]
        top = np.argsort(-(ex > EXCESS[0]).mean(0))[:4]
        return {"frames": len(mp),
                "frac_over_1deg": float((worst > EXCESS[0]).mean()),
                "frac_over_6deg": float((worst > EXCESS[1]).mean()),
                "max_excess_deg": float(np.degrees(worst.max())),
                "worst_params": {names[k]: round(float((ex[:, k] > EXCESS[0]).mean()), 3) for k in top if ex[:, k].max() > EXCESS[0]},
                "ik_residual_deg_median": float(np.median(res)),
                "ik_residual_deg_p99": float(np.percentile(res, 99))}


def stress(times, world, image, vis, present, rng):
    """Stressed copies of a session: name -> (world, image, vis, present)."""
    T = len(times)
    out = {"recorded": (world, image, vis, present)}
    p = present.copy()
    for start in np.linspace(0.1, 0.8, 3) * times[-1]:
        p[(times >= start) & (times < start + 2.0)] = False
    out["tracking loss"] = (world, image, vis, p)
    out["low confidence"] = (world, image, vis * 0.3, present)
    out["extra noise"] = (world + rng.normal(0, 0.02, world.shape), image + rng.normal(0, 0.01, image.shape), vis, present)
    w, im, v = world.copy(), image.copy(), vis.copy()
    hands = [15, 16, 17, 18, 19, 20, 21, 22]
    im[:, hands, 1] = 1.3 + rng.normal(0, 0.05, (T, len(hands)))
    w[:, hands] += np.cumsum(rng.normal(0, 0.01, (T, len(hands), 3)), 0)
    v[:, hands] = 0.03
    out["arms out of frame"] = (w, im, v, present)
    w, im = world.copy(), image.copy()
    for k in rng.choice(T, max(1, T // 20), replace=False):
        j = rng.choice(33, 6, replace=False)
        w[k, j] += rng.normal(0, 0.3, (6, 3))
        im[k, j] += rng.normal(0, 0.2, (6, 3))
    out["glitches"] = (w, im, vis, present)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("session")
    ap.add_argument("--gt-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = Runner(a.ckpt)
    lc = LimitChecker(r)
    rng = np.random.default_rng(a.seed)
    rows = []
    if a.gt_samples:
        from .dataset import fixed_set
        gt = fixed_set("val", a.gt_samples, seed=a.seed)["rot"].numpy().astype(np.float64)
        rows.append(("synthetic ground truth (checker validation)", lc.report(*lc.fit(r.mhr_globals(gt.reshape(-1, *gt.shape[2:]))))))
    for name, obs in stress(*read_session(a.session), rng).items():
        times = read_session(a.session)[0]
        rot, _ = r.predict(times, *obs)
        rows.append((f"model: {name}", lc.report(*lc.fit(r.mhr_globals(rot)))))
    print(f"{'condition':46s} {'frames':>6s} {'>1deg':>7s} {'>6deg':>7s} {'max':>6s} {'IK res med/p99':>15s}  worst parameters (frac > 1 deg)")
    for name, m in rows:
        print(f"{name:46s} {m['frames']:6d} {m['frac_over_1deg']:7.3f} {m['frac_over_6deg']:7.3f} {m['max_excess_deg']:6.1f}"
              f" {m['ik_residual_deg_median']:7.1f}/{m['ik_residual_deg_p99']:<7.1f}  {m['worst_params']}")


if __name__ == "__main__":
    main()
