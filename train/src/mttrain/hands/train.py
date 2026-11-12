"""Train the hand model on procedural finger motion with synthetic MediaPipe observations (GPU).

    uv run python -m mttrain.hands.train --name h1 --calib data/m4/calib/hands1.json [--steps 20000]

Runs in data/m4/runs/<name>: config.json, log.jsonl, best.pt / last.pt. Each sample: a random window
of a pool sequence (`pool`), truth hand points by GPU forward kinematics (MHR's left hand), canonical
form, synthetic observations (`observe.Observer`); target = the finger parameters at the newest frame.
Losses: Laplace NLL on the parameters (with the predicted scale) + L1 on the canonical points of the
predicted hand (FK), the quantity the evaluation scores. Validation: the val pool, fixed noise.
"""

import argparse
import json
import time

import numpy as np
import torch

from mtcore.mhr import load
from mtcore.paths import DATA

from ..gpugen.fk import MhrFK
from .model import HISTORY, OUT_IDX, HandNet, frame_input
from .observe import Observer, canonical_t
from .pool import POOL
from .skeleton import HandFK

RUNS = DATA / "m4" / "runs"


class Batches:
    def __init__(self, split, calib, device, seed):
        self.q = torch.tensor(np.load(POOL / f"{split}.npz")["q"], device=device)  # (N, T, 27)
        hfk = HandFK()
        self.idx = torch.tensor(hfk.idx["l"], device=device)
        self.n_param = len(hfk.pn)
        self.mp21 = torch.tensor(hfk.mp21["l"], device=device)
        self.fk = MhrFK(hfk.ch, keep=hfk.mp21["l"].tolist()).to(device)
        self.obs = Observer(calib, device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def points(self, q):
        """Finger parameters (..., 27) -> canonical points (..., 21, 3) of the left hand."""
        full = torch.zeros(q.shape[:-1] + (self.n_param,), device=q.device, dtype=q.dtype)
        full[..., self.idx] = q
        st = self.fk(full.reshape(-1, self.n_param))
        P = st[:, self.mp21, :3].reshape(q.shape[:-1] + (21, 3)) / 100.0
        return canonical_t(P)

    def sample(self, B):
        N, T, _ = self.q.shape
        s = torch.randint(0, N, (B,), device=self.device, generator=self.gen)
        t = torch.randint(HISTORY - 1, T, (B,), device=self.device, generator=self.gen)
        frames = t[:, None] - torch.arange(HISTORY, device=self.device)[None]  # newest first
        q = self.q[s[:, None], frames]  # (B, H, 27)
        truth = self.points(q)
        obs, lost = self.obs(truth.flip(1), self.gen)  # observer runs forward in time
        obs, lost = obs.flip(1), lost.flip(1)
        return frame_input(obs, lost), q[:, 0], truth[:, 0]


def losses(model, batches, x, q, truth):
    pred, log_s = model(x)
    target = q[:, OUT_IDX]
    nll = (torch.abs(pred - target) * torch.exp(-log_s) + log_s).mean()
    full = q.clone()
    full[:, OUT_IDX] = pred
    pts = (batches.points(full) - truth).abs().sum(-1).mean()
    ang = torch.abs(pred - target).mean()
    return nll, pts, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-points", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    train, val = Batches("train", a.calib, dev, a.seed), Batches("val", a.calib, dev, 12345)
    vx = [val.sample(a.batch) for _ in range(8)]
    model = HandNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    best, t0 = float("inf"), time.time()
    log = open(run / "log.jsonl", "a")
    for step in range(1, a.steps + 1):
        model.train()
        nll, pts, ang = losses(model, train, *train.sample(a.batch))
        loss = nll + a.w_points * pts
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                v = np.mean([[float(z) for z in losses(model, val, *b)] for b in vx], 0)
            rec = {"step": step, "train_nll": float(nll), "train_pts": float(pts), "val_nll": v[0], "val_pts": v[1],
                   "val_ang_deg": float(np.degrees(v[2])), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "step": step}, run / "last.pt")
            if v[1] < best:
                best = v[1]
                torch.save({"model": model.state_dict(), "step": step}, run / "best.pt")


if __name__ == "__main__":
    main()
