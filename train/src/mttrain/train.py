"""Train the M1 body lifter.

    uv run python -m mttrain.train --name first [--steps 40000] [--batch 256] [--device cuda:1] [--gen gpu|cpu]

Training data comes from the batched GPU generator (`mttrain.gpugen`) on the training device by
default; `--gen cpu` uses the numpy generator in DataLoader worker processes (heavy on the CPU: run
it under tools/guarded-run.sh). Validation always uses the numpy generator (fixed set).

Writes data/m1/runs/<name>/: config.json, log.jsonl (train losses and validation metrics),
best.pt / last.pt (model, limit box, config). CMU-derived: local only (license register).
"""

import argparse
import json
import math
import os
import time

import torch

from mtcore.paths import DATA

from .dataset import SampleStream, fixed_set
from .losses import limit_box, losses, metrics
from .model import Lifter

WEIGHTS = {"rot": 1.0, "pos": 5.0, "vel": 2.0, "lim": 10.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--gen", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--gen-chunk", type=int, default=4, help="GPU generator: batches generated per call")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = DATA / "m1" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = vars(a) | {"weights": WEIGHTS, "fits": os.environ.get("MT_FITS", "fits")}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    t0 = time.time()
    val = {k: v.to(dev) for k, v in fixed_set("val", a.val).items()}
    ref = fixed_set("train", 4096, seed=999)
    box = limit_box(ref["rot"].flatten(0, 1)).to(dev)
    print(f"validation set and limit box ready in {time.time() - t0:.0f}s", flush=True)

    model = Lifter().to(dev)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    if a.gen == "gpu":
        from .gpugen.generator import GpuGenerator
        torch.set_num_threads(2)
        gg = GpuGenerator("train", dev, seed=a.seed)

        def gpu_batches():
            while True:
                big = gg.sample(a.batch * a.gen_chunk)
                for i in range(a.gen_chunk):
                    yield {k: v[i * a.batch:(i + 1) * a.batch] for k, v in big.items()}
        loader = gpu_batches()
    else:
        loader = torch.utils.data.DataLoader(SampleStream("train", seed=a.seed), batch_size=a.batch,
                                             num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    log = open(out / "log.jsonl", "a")
    best = float("inf")
    run, tl = {}, time.time()
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        rot, ls = model(batch["x"])
        loss, parts = losses(rot, ls, batch, box, WEIGHTS)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = run.get(k, 0.0) + v
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vr, vl = [], 0.0
                for i in range(0, a.val, 1024):
                    vb = {k: v[i:i + 1024] for k, v in val.items()}
                    r, s = model(vb["x"])
                    vl += losses(r, s, vb, box, WEIGHTS)[0].item() * len(vb["x"])
                    vr.append(r)
                m = metrics(torch.cat(vr), val, box) | {"val_loss": vl / a.val}
            model.train()
            rec = {"step": step, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0),
                   "samples_per_s": round(a.eval_every * a.batch / (time.time() - tl))} \
                | {f"train_{k}": v / a.eval_every for k, v in run.items()} | m
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            ckpt = {"model": model.state_dict(), "box": box.cpu(), "config": cfg, "step": step, "metrics": m}
            torch.save(ckpt, out / "last.pt")
            if m["val_loss"] < best:
                best = m["val_loss"]
                torch.save(ckpt, out / "best.pt")
            run, tl = {}, time.time()
        if step >= a.steps:
            break


if __name__ == "__main__":
    main()
