"""Batched training samples on the GPU: the torch counterpart of mttrain.dataset.Generator.sample,
one whole batch per call.

Differences from the numpy generator, all distribution-level:
- The source crop has a fixed length (the longest any frame rate needs, 3.2 s at 8 fps), so clips
  shorter than that (a handful of the 90-96-frame ones) are not used; detector frames still span
  1.6 s + K frames + 0.6 s per sequence as before, padded to a common length.
- Random streams differ (torch generator), and a few per-template draws are shared per arm rather
  than per template occurrence (see motion.py).

The choreography pool (dataset.source_table) is drawn like the clips; its sequences are already
seated, so the seater leaves them alone, and they count as seated.
"""

import math

import numpy as np
import torch

from mtcore.landmarks import LandmarkModel
from mtcore.mhr import limits, load, skeleton_state

from ..dataset import SRC_FPS, source_table
from mtcore.features import OFFSETS
from . import features as GF
from .fk import MhrFK
from .geom import (HeadRig, Landmarks, Style, Targets, flat_image_z, hand_lock, head_lock, head_noise, project,
                   sample_desk_camera, shoulder_distance, world_landmarks)
from .motion import Seater
from .observe import observe, sample_config


class GpuGenerator:
    def __init__(self, split, device, K=8, p_seated=0.75, seed=0, min_fps=8.0, max_fps=30.0, p_choreo=0.2):
        self.dev = torch.device(device)
        self.K, self.p_seated, self.min_fps, self.max_fps = K, p_seated, min_fps, max_fps
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        ch = load()
        pn = list(ch.parameter_transform.names)
        jn = list(ch.skeleton.joint_names)
        rest = skeleton_state(ch, np.zeros(len(pn)))
        lmodel = LandmarkModel(ch, rest)
        self.lm = Landmarks(lmodel).to(self.dev)
        self.targets = Targets(jn, rest).to(self.dev)
        self.head_rig = HeadRig(jn, rest, lmodel).to(self.dev)
        keep = sorted(set(self.lm.joint.tolist()) | set(self.targets.idx.tolist()))
        self.fk = MhrFK(ch, keep).to(self.dev)
        self.seater = Seater(pn, limits()).to(self.dev)
        self.max_dur = OFFSETS[-1] + K / min_fps + 0.6
        self.N = int(math.ceil(self.max_dur * SRC_FPS)) + 1  # source frames per crop
        self.Tmax = int(math.ceil(self.max_dur * max_fps)) + 2  # detector frames, padded
        rows = [r for r in source_table(split, p_choreo) if len(r[1]) >= self.N]
        mps = {k: mp for k, mp, _, _ in rows}
        self.keys = [k for k, _, _, _ in rows]
        w = torch.tensor([p for _, _, p, _ in rows], dtype=torch.float32)
        self.p = (w / w.sum()).to(self.dev)
        self.is_choreo = torch.tensor([c for _, _, _, c in rows], device=self.dev)
        lens = torch.tensor([len(mps[k]) for k in self.keys])
        self.start = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]]).to(self.dev)
        self.lens = lens.to(self.dev)
        self.mp = torch.tensor(np.concatenate([mps[k] for k in self.keys]), dtype=torch.float32, device=self.dev)

    def _u(self, B, lo, hi):
        return lo + (hi - lo) * torch.rand(B, generator=self.gen, device=self.dev)

    def frame_times(self, fps, dur, drop_rate=0.03):
        """Padded detector times (B, Tmax) (+inf after the last frame) and valid (B, Tmax)."""
        B, dev = len(fps), self.dev
        k = torch.arange(self.Tmax, device=dev, dtype=torch.float32)[None]
        nominal = k / fps[:, None]
        t = nominal + 0.004 * torch.randn(B, self.Tmax, generator=self.gen, device=dev)
        t = torch.minimum(t.clamp_min(0), (dur - 1e-6)[:, None])
        keep = (torch.rand(B, self.Tmax, generator=self.gen, device=dev) >= drop_rate) & (nominal < dur[:, None])
        keep[:, 0] = True
        t = torch.where(keep, t, torch.full_like(t, math.inf)).sort(-1).values
        return t, torch.isfinite(t)

    @torch.no_grad()
    def sample(self, B):
        fps, mp, seated = self.crop(B)
        out = self.observe(mp, fps)
        return dict(x=out["x"], rot=out["rot"], pos=out["pos"], off=out["off"], cof=out["cof"], dt=out["dt"],
                    seated=seated.float())

    def crop(self, B, fps=None, smooth_life=False):
        """Source crops: detector rates (B,) (random unless given), seated MHR parameters (B, N, P) at SRC_FPS,
        seated (B,). `smooth_life`: low-pass the seater's idle signals (jitter-free targets, merger)."""
        dev = self.dev
        f = torch.where(torch.rand(B, generator=self.gen, device=dev) < 0.5, torch.full((B,), 10.0, device=dev),
                        self._u(B, self.min_fps, self.max_fps))
        fps = f if fps is None else torch.full((B,), float(fps), device=dev)
        clip = torch.multinomial(self.p, B, replacement=True, generator=self.gen)
        s0 = torch.floor(torch.rand(B, generator=self.gen, device=dev) * (self.lens[clip] - self.N + 1)).long()
        rows = self.start[clip, None] + s0[:, None] + torch.arange(self.N, device=dev)[None]
        seated = torch.rand(B, generator=self.gen, device=dev) < self.p_seated
        choreo = self.is_choreo[clip]
        mp = self.seater(self.gen, self.mp[rows], seated & ~choreo, SRC_FPS, smooth_life=smooth_life)
        return fps, mp, seated | choreo

    @torch.no_grad()
    def observe(self, mp, fps, drop_rate=0.03, push=None):
        """Source crops (B, N, P) at SRC_FPS, detector rates (B,) -> model input x (B, K, ...) for the last K
        detector frames, their targets, and the detector frames (times, valid, states, target frame indices) and
        camera (R, t, fx) for other observers (merger). `push`: optional function on the 3D landmarks (B, T, 33, 3)
        metres before they are observed (merger: observed hands pushed into the body)."""
        dev, K = self.dev, self.K
        B = mp.shape[0]
        dur = OFFSETS[-1] + K / fps + 0.6
        times, valid = self.frame_times(fps, dur, drop_rate)
        tz = torch.where(valid, times, torch.zeros_like(times))
        src = torch.round(tz * SRC_FPS).long().clamp(max=self.N - 1)
        states = self.fk(mp.gather(1, src[..., None].expand(B, self.Tmax, mp.shape[-1])))  # (B, T, J, 8)
        X = self.lm(states) / 100.0
        if push is not None:
            X = push(X)
        style = Style.sample(B, self.gen, dev)
        dt = torch.diff(tz, dim=1, prepend=(tz[:, :1] - 1.0 / fps[:, None]))
        dt = torch.where(valid, dt, (1.0 / fps)[:, None].expand_as(dt))
        # MediaPipe's partial head and wrist-bend response, separately for the image and world landmarks
        delta, C = self.head_rig(states, X)
        Xs = {}
        for space in ("image", "world"):
            Y = head_lock(X, delta, C, style.head_gain[space], torch.deg2rad(style.head_nod_deg[space]),
                          head_noise(self.gen, dt, style.head_noise_deg[space]))
            hn = torch.stack([head_noise(self.gen, dt, style.hand_noise_deg[space]) for _ in range(2)], 2)
            Xs[space] = style.placement(hand_lock(Y, style.hand_gain[space], hn))
        Xi, Xw = Xs["image"], Xs["world"]
        Xn = torch.where(valid[..., None, None], Xi, torch.full_like(Xi, math.nan))
        R, t, fx = sample_desk_camera(Xn, self.gen)
        image = project(Xi, R, t, fx)
        Xwn = torch.where(valid[..., None, None], Xw, torch.full_like(Xw, math.nan))
        flat = style.flatten(world_landmarks(Xw, R, t), shoulder_distance(Xwn, R, t))
        image[..., 2] = flat_image_z(Xi, R, t, fx, flat)
        world = style.world(flat)
        W, Im, vis, present = observe(self.gen, world, image, dt, sample_config(B, self.gen, dev))
        Im[..., 2] = Im[..., 2] * style.image_z[:, None, None]
        present = present & valid

        n_valid = valid.sum(1)
        tgt = (n_valid[:, None] - K + torch.arange(K, device=dev)[None]).clamp_min(0)
        st_t = states.gather(1, tgt[..., None, None].expand(B, K, *states.shape[2:]))
        rot, pos, off = self.targets(st_t, R)
        cof = self.targets.contact_offsets(st_t, R, self.lm(st_t))
        feats = GF.frame_features(times, W, Im, vis, present, valid, style.head3d)
        idx, age = GF.history_indices(times, tgt)
        # session start: no history before a random point, for 10% of samples
        cut_on = torch.rand(B, generator=self.gen, device=dev) < 0.1
        t_first = tz[:, 0]
        t_tgt0 = tz.gather(1, tgt[:, :1])[:, 0]
        cut = t_first + (t_tgt0 - t_first) * torch.rand(B, generator=self.gen, device=dev)
        picked = tz.gather(1, idx.clamp_min(0).reshape(B, -1)).reshape(idx.shape)
        idx = torch.where(cut_on[:, None, None] & (picked < cut[:, None, None]), torch.full_like(idx, -1), idx)
        x = GF.model_input(feats, idx, age)
        prev = tz.gather(1, (tgt[:, :1] - 1).clamp_min(0))
        tt = tz.gather(1, tgt)
        dt_out = torch.diff(tt, dim=1, prepend=prev)
        return dict(x=x, rot=rot, pos=pos, off=off, cof=cof, dt=dt_out, times=times, valid=valid, states=states,
                    tgt=tgt, R=R, t=t, fx=fx)
