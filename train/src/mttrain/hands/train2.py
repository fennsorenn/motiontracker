"""Train the pair hand model (v2) on the posed two-hand pool (`pool2`) with synthetic observations (`observe2`).

    uv run python -m mttrain.hands.train2 --name p1 --calib data/m4/calib/hands1.json [--steps 30000]

Runs in data/m4/runs/<name>. Per sample: a window of HISTORY + 1 frames, a random camera, observations of
both hands with flips, swaps, doubled and lost hands; the model runs for both hands (own = left slot, and
the mirrored scene with own = right slot) at the newest frame and one frame earlier (velocity loss).
Targets follow hand identity, not slots: a swapped detection must still give the true hand.
Losses: fingers (Laplace NLL + canonical point L1 via GPU FK + velocity), orientation (geodesic angle with a
Laplace scale + velocity), position (wrist in the image and log palm length, in palm lengths). Hands out of view
count too (v3: the NLLs fully, the rest by OUT_W), so the model learns their probable pose and its scales grow
there; `--scores` adds the detector's per-slot confidence as input (observe2, scorecal).
"""

import argparse
import json
import time

import numpy as np
import torch

from mtcore.paths import DATA

from ..gpugen.fk import MhrFK
from .model import OUT_IDX
from .model2 import HISTORY, PairNet, scenes
from .observe import canonical_t
from .observe2 import MIRROR_X, PairObserver, palm_frame_t
from .pool2 import POOL2
from .skeleton import HandFK

RUNS = DATA / "m4" / "runs"


class Batches:
    def __init__(self, split, calib, device, seed, pool=POOL2, scores=None):
        z = np.load(pool / f"{split}.npz")
        self.hands = torch.tensor(z["hands"].astype(np.float32), device=device)  # (N, T, 2, 21, 3)
        self.body = torch.tensor(z["body"].astype(np.float32), device=device)  # (N, T, 33, 3)
        self.q = torch.tensor(z["q"].astype(np.float32), device=device)  # (N, T, 2, 27)
        hfk = HandFK()
        self.idx = torch.tensor(hfk.idx["l"], device=device)
        self.n_param = len(hfk.pn)
        self.mp21 = torch.tensor(hfk.mp21["l"], device=device)
        self.fk = MhrFK(hfk.ch, keep=hfk.mp21["l"].tolist()).to(device)
        self.obs = PairObserver(calib, device, scores)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    def points(self, q):
        full = torch.zeros(q.shape[:-1] + (self.n_param,), device=q.device, dtype=q.dtype)
        full[..., self.idx] = q
        st = self.fk(full.reshape(-1, self.n_param))
        return canonical_t(st[:, self.mp21, :3].reshape(q.shape[:-1] + (21, 3)) / 100.0)

    def sample(self, B):
        N, T = self.hands.shape[:2]
        F = HISTORY + 1
        s = torch.randint(0, N, (B,), device=self.device, generator=self.gen)
        t = torch.randint(F - 1, T, (B,), device=self.device, generator=self.gen)
        frames = t[:, None] - torch.arange(F - 1, -1, -1, device=self.device)[None]  # oldest first
        obs, truth = self.obs(self.hands[s[:, None], frames], self.body[s], self.gen)
        q = self.q[s[:, None], frames]  # (B, F, 2, 27)
        flip = lambda x: x.flip(1)  # newest first for the model
        return ({k: flip(v) for k, v in obs.items()}, {k: flip(v) for k, v in truth.items() if v.dim() >= 2},
                flip(q))


def targets(truth, q):
    """Per scene (2B rows) at frames 0 (newest) and 1: fingers (2B, 2, 27), rotation (2B, 2, 3, 3), wrist (2B, 2, 2),
    log size (2B, 2), inside (2B, 2)."""
    img = truth["img"][:, :2]  # (B, 2 frames, 2 slots, 21, 3)
    left, right = img[:, :, 0], img[:, :, 1] * MIRROR_X.to(img)
    pts = torch.cat([left, right], 0)
    _, size = palm_frame_t(pts)
    # orientation target: the true 3D palm frame in the camera frame (right hands mirrored), not the image-point
    # frame (perspective x/y and relative depth disagree in scale: 13 deg off the true orientation)
    cam = truth["cam"][:, :2]
    R, _ = palm_frame_t(torch.cat([cam[:, :, 0], cam[:, :, 1] * MIRROR_X.to(cam)], 0))
    inside = torch.cat([truth["inside"][:, :2, 0], truth["inside"][:, :2, 1]], 0)
    return {"q": torch.cat([q[:, :2, 0], q[:, :2, 1]], 0), "rot": R, "wrist": pts[..., 0, :2],
            "log_size": torch.log(size.clamp_min(1e-4)), "inside": inside, "size": size}


def geodesic(Ra, Rb):
    c = ((Ra * Rb).sum((-1, -2)) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


OUT_W = 0.3  # out-of-view hands: weight of the point, velocity and position losses (the NLLs take them fully)


def losses(model, batches, obs, truth, q):
    own, other = scenes(obs["img"], obs["canon"], obs["lost"], obs.get("score"))  # (2B, F, D) newest first
    tg = targets(truth, q)
    out0 = model(own[:, :HISTORY], other[:, :HISTORY])
    out1 = model(own[:, 1:], other[:, 1:])
    inside = tg["inside"][:, 0].float()
    # v3 (owner, 2026-11-13): hands out of view are scored too, so the model learns their probable pose (a hand
    # that left at the bottom hangs or rests) and its Laplace scales grow there (the merger's confidence)
    w_all = torch.ones_like(inside)
    w_soft = inside + OUT_W * (1 - inside)
    norm = lambda x, w: (x * w).sum() / w.sum().clamp_min(1.0)
    tq = tg["q"][:, 0, OUT_IDX]
    l_s = out0["fingers_log_s"]
    nll = norm((torch.abs(out0["fingers"] - tq) * torch.exp(-l_s) + l_s).mean(-1), w_all)
    full = tg["q"][:, 0].clone()
    full[:, OUT_IDX] = out0["fingers"]
    pts = norm((batches.points(full) - batches.points(tg["q"][:, 0])).abs().sum(-1).mean(-1), w_soft)
    vel = norm(torch.abs((out0["fingers"] - out1["fingers"]) - (tq - tg["q"][:, 1, OUT_IDX])).mean(-1), w_soft)
    ang = geodesic(out0["rot"], tg["rot"][:, 0])
    rot_nll = norm(ang * torch.exp(-out0["rot_log_s"]) + out0["rot_log_s"], w_all)
    ang1 = geodesic(out0["rot"] @ out1["rot"].transpose(-1, -2), tg["rot"][:, 0] @ tg["rot"][:, 1].transpose(-1, -2))
    rot_vel = norm(ang1, w_soft)
    pos = norm((out0["pos"] - tg["wrist"][:, 0]).abs().sum(-1) / tg["size"][:, 0].clamp_min(1e-3)
               + (out0["log_size"] - tg["log_size"][:, 0]).abs(), w_soft)
    fe = torch.rad2deg(torch.abs(out0["fingers"] - tq).mean(-1))
    return {"nll": nll, "pts": pts, "vel": vel, "rot_nll": rot_nll, "rot_vel": rot_vel, "pos": pos,
            "rot_deg": norm(torch.rad2deg(ang), inside), "fing_deg": norm(fe, inside),
            "rot_deg_out": norm(torch.rad2deg(ang), 1 - inside), "fing_deg_out": norm(fe, 1 - inside),
            "rot_s_deg": norm(torch.rad2deg(torch.exp(out0["rot_log_s"])), inside),
            "rot_s_deg_out": norm(torch.rad2deg(torch.exp(out0["rot_log_s"])), 1 - inside)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-points", type=float, default=2.0)
    ap.add_argument("--w-vel", type=float, default=2.0)
    ap.add_argument("--w-rot", type=float, default=1.0)
    ap.add_argument("--w-pos", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default=str(POOL2))
    ap.add_argument("--scores", help="detector score calibration (scorecal) -> the model's score input (v3)")
    a = ap.parse_args()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    from pathlib import Path
    train = Batches("train", a.calib, dev, a.seed, Path(a.pool), a.scores)
    val = Batches("val", a.calib, dev, 12345, Path(a.pool), a.scores)
    vb = [val.sample(a.batch) for _ in range(6)]
    model = PairNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    best, t0 = float("inf"), time.time()
    log = open(run / "log.jsonl", "a")
    for step in range(1, a.steps + 1):
        model.train()
        L = losses(model, train, *train.sample(a.batch))
        loss = L["nll"] + a.w_points * L["pts"] + a.w_vel * (L["vel"] + 0.5 * L["rot_vel"]) + a.w_rot * L["rot_nll"] + a.w_pos * L["pos"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                V = [losses(model, val, *b) for b in vb]
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1)}
            rec.update({f"val_{k}": round(float(np.mean([float(v[k]) for v in V])), 4) for k in V[0]})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "step": step}, run / "last.pt")
            score = rec["val_pts"] + rec["val_rot_deg"] / 100
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "step": step}, run / "best.pt")


if __name__ == "__main__":
    main()
