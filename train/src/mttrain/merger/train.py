"""Train the merger (plan `dev-notes/plans/m4-merger.md`) on `MergerBatches`, part models frozen.

    uv run python -m mttrain.merger.train --name mg1 [--steps 20000] [--jitter-deg 0.75] [--resume]

Runs in data/merger/runs/<name>. Per sample the merger runs at the newest three frames (velocity and acceleration
losses). Losses: the body model's (`mttrain.losses`: rotation with Laplace scales, FK positions, velocity, joint
limits with the body model's box, contacts), global angle of wrists and head, fingers (Laplace NLL on the
parameters, in and out of view), and acceleration of FK positions and fingers against the truth's (smooth output
without lag: the truth keeps its sharp moves). Validation: fixed batches from the val split, with the base pose and
the body model alone for comparison. Resumable: every evaluation writes last.pt with optimizer, scheduler and
generator states; `--resume` continues from it.
"""

import argparse
import json
import time

import torch

from ..hands.model import OUT_IDX
from ..losses import POS_IDX, losses as body_losses, metrics as body_metrics
from ..skeleton import OUT_JOINTS, fk
from .batch import MERGER, MergerBatches, geodesic
from .model import Merger, apply, frame_inputs

RUNS = MERGER / "runs"
GLOBAL_J = [OUT_JOINTS.index(j) for j in ("l_wrist", "r_wrist", "c_head")]
WEIGHTS = {"rot": 1.0, "pos": 5.0, "vel": 2.0, "lim": 1.0, "con": 2.0}
NEWEST = (-1, -2, -3)
SCORE = ("glob_l_wrist", "glob_r_wrist", "glob_c_head", "deg_forearm", "deg_head", "fing_deg_in")  # best.pt: lowest sum
M1_RUN = MERGER.parent / "m1" / "runs" / "m1-v10" / "best.pt"


def run(model, b):
    feats = frame_inputs(b)
    K = feats.shape[1]
    outs = [apply(model, b, feats, K + n) for n in NEWEST]  # newest, then one and two frames earlier
    L = torch.stack([o[0] for o in outs[::-1]], 1)  # (B, 3, NJ, 3, 3) oldest first
    ls = torch.stack([o[1] for o in outs[::-1]], 1)
    f = torch.stack([o[2] for o in outs[::-1]], 1)  # (B, 3, 2, NF)
    fs = torch.stack([o[3] for o in outs[::-1]], 1)
    return L, ls, f, fs


def last3(x):
    return x[:, -3:]


def loss_fn(b, L, ls, f, fs, box, w):
    tg = {k: last3(v) for k, v in b["target"].items() if k not in ("seated",)}
    tg["seated"] = b["target"]["seated"]
    total, parts = body_losses(L, ls, tg, box, {**WEIGHTS, "con": w.get("con", WEIGHTS["con"])})
    G, P = fk(L, tg["off"])
    Gt, Pt = fk(tg["rot"], tg["off"])
    l_glob = torch.rad2deg(geodesic(G[..., GLOBAL_J, :, :], Gt[..., GLOBAL_J, :, :])).mean() / 10.0
    tf = tg["fingers"][..., OUT_IDX]  # (B, 3, 2, NF)
    l_f = (torch.abs(f - tf) * torch.exp(-fs) + fs).mean()
    acc = lambda x: x[:, 2] - 2 * x[:, 1] + x[:, 0]
    l_acc = (acc(P[..., POS_IDX, :]) - acc(Pt[..., POS_IDX, :])).norm(dim=-1).mean() * 10 + \
        torch.abs(acc(f) - acc(tf)).mean()
    total = total + w["glob"] * l_glob + w["fing"] * l_f + w["acc"] * l_acc
    parts.update(glob=l_glob.item(), fing=l_f.item(), acc=l_acc.item())
    return total, parts


@torch.no_grad()
def evaluate(model, val, box):
    """Mean over the validation batches: merger, base pose and body model alone."""
    acc = {}
    n = 0
    for b in val:
        L, ls, f, fs = run(model, b)
        tg = {k: last3(v) for k, v in b["target"].items() if k != "seated"}
        tg["seated"] = b["target"]["seated"]
        rows = {"mg": (L, f), "base": (last3(b["base"]), torch.stack([last3(b["hand"][s]["fingers"]) for s in "lr"], 2)),
                "m1": (last3(b["body"]["rot"]), None)}
        G_t, P_t = fk(tg["rot"], tg["off"])
        ins = tg["inside"]  # (B, 3, 2)
        for name, (R, fing) in rows.items():
            m = body_metrics(R, tg, box)
            G, P = fk(R, tg["off"])
            ang = torch.rad2deg(geodesic(G, G_t))
            for j in ("l_wrist", "r_wrist", "c_head", "l_lowarm"):
                m[f"glob_{j}"] = ang[..., OUT_JOINTS.index(j)].mean().item()
            a = lambda x: x[:, 2] - 2 * x[:, 1] + x[:, 0]
            m["acc_cm"] = ((a(P) - a(P_t)).norm(dim=-1).mean() * 100).item()
            m["acc_abs_cm"] = (a(P).norm(dim=-1).mean() * 100).item()  # output jerkiness (truth: acc_true_cm)
            if fing is not None:
                fe = torch.rad2deg(torch.abs(fing - tg["fingers"][..., OUT_IDX]).mean(-1))  # (B, 3, 2)
                m["fing_deg_in"] = fe[ins].mean().item()
                m["fing_deg_out"] = fe[~ins].mean().item() if (~ins).any() else float("nan")
            for k, v in m.items():
                acc[f"{name}_{k}"] = acc.get(f"{name}_{k}", 0.0) + v
        acc["acc_true_cm"] = acc.get("acc_true_cm", 0.0) + (
            (P_t[:, 2] - 2 * P_t[:, 1] + P_t[:, 0]).norm(dim=-1).mean() * 100).item()
        n += 1
    return {k: round(v / n, 4) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--jitter-deg", type=float, default=0.75,
                    help="extra input jitter on the part rotations (synthetic part outputs already jitter about as much as real ones)")
    ap.add_argument("--w-glob", type=float, default=1.0)
    ap.add_argument("--w-fing", type=float, default=1.0)
    ap.add_argument("--w-acc", type=float, default=2.0)
    ap.add_argument("--w-con", type=float, default=WEIGHTS["con"], help="contact loss (hand-hand, hand-face points)")
    ap.add_argument("--val", type=int, default=16, help="validation batches (of --batch / 2)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-gap", action="store_true", help="no body model sim-to-real gap noise (batch.degrade)")
    ap.add_argument("--pool", default=str(MERGER / "pool"), help="merger pool directory (train.npz, val.npz)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from pathlib import Path
    vb = MergerBatches("val", dev, seed=1000, pool=Path(a.pool))
    val = [vb.sample(a.batch // 2, jitter_deg=0.0, gap=not a.no_gap) for _ in range(a.val)]
    del vb
    tb = MergerBatches("train", dev, seed=a.seed, pool=Path(a.pool))
    box = torch.load(M1_RUN, map_location="cpu", weights_only=False)["box"].to(dev)  # the body model's limit box
    print(f"validation and batches ready in {time.time() - t0:.0f}s", flush=True)
    model = Merger().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + torch.cos(torch.tensor(min(s, a.steps) / a.steps * 3.14159)).item()))
    w = {"glob": a.w_glob, "fing": a.w_fing, "acc": a.w_acc, "con": a.w_con}
    cfg = vars(a)
    start, best = 0, float("inf")
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        tb.gen.set_state(ck["gen"].to(tb.gen.device) if ck["gen"].device.type != "cpu" else ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        start, best = ck["step"], ck.get("best", best)
        print(f"resumed at step {start}", flush=True)
    else:
        rec = {"step": 0, **{k: v for k, v in evaluate(model, val, box).items() if k.startswith(("base", "m1", "acc_true"))}}
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as fl:
            fl.write(json.dumps(rec) + "\n")
    model.train()
    for step in range(start + 1, a.steps + 1):
        b = tb.sample(a.batch, jitter_deg=a.jitter_deg, gap=not a.no_gap)
        L, ls, f, fs = run(model, b)
        loss, parts = loss_fn(b, L, ls, f, fs, box, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            m = evaluate(model, val, box)
            model.train()
            score = sum(m[f"mg_{k}"] for k in SCORE)
            rec = {"step": step, "sec": round(time.time() - t0), "loss": round(loss.item(), 4),
                   **{k: round(v, 4) for k, v in parts.items()}, **{k: v for k, v in m.items() if k.startswith(("mg", "acc_true"))}}
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a") as fl:
                fl.write(json.dumps(rec) + "\n")
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "gen": tb.gen.get_state().cpu(), "torch_rng": torch.get_rng_state(), "step": step, "config": cfg,
                  "metrics": m, "best": min(best, score)}
            torch.save(ck, out / "last.tmp.pt")
            (out / "last.tmp.pt").replace(out / "last.pt")
            if score < best:
                best = score
                torch.save(ck, out / "best.tmp.pt")
                (out / "best.tmp.pt").replace(out / "best.pt")


if __name__ == "__main__":
    main()
