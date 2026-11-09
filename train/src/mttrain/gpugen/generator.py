"""Batched training samples on the GPU: the torch counterpart of mttrain.dataset.Generator.sample,
one whole batch per call.

Differences from the numpy generator, all distribution-level:
- The source crop has a fixed length (the longest any frame rate needs, 3.2 s at 8 fps), so clips
  shorter than that (a handful of the 90-96-frame ones) are not used; detector frames still span
  1.6 s + K frames + 0.6 s per sequence as before, padded to a common length.
- Random streams differ (torch generator), and a few per-template draws are shared per arm rather
  than per template occurrence (see motion.py).
"""

import math

import numpy as np
import torch

from mtcore.landmarks import LandmarkModel
from mtcore.mhr import limits, load, skeleton_state

from ..dataset import FITS, SRC_FPS, clip_table
from ..features import OFFSETS
from . import features as GF
from .fk import MhrFK
from .geom import Landmarks, Style, Targets, flat_image_z, project, sample_desk_camera, shoulder_distance, world_landmarks
from .motion import Seater
from .observe import observe, sample_config


class GpuGenerator:
    def __init__(self, split, device, K=8, p_seated=0.75, seed=0, min_fps=8.0, max_fps=30.0):
        self.dev = torch.device(device)
        self.K, self.p_seated, self.min_fps, self.max_fps = K, p_seated, min_fps, max_fps
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        ch = load()
        pn = list(ch.parameter_transform.names)
        jn = list(ch.skeleton.joint_names)
        rest = skeleton_state(ch, np.zeros(len(pn)))
        self.lm = Landmarks(LandmarkModel(ch, rest)).to(self.dev)
        self.targets = Targets(jn, rest).to(self.dev)
        keep = sorted(set(self.lm.joint.tolist()) | set(self.targets.idx.tolist()))
        self.fk = MhrFK(ch, keep).to(self.dev)
        self.seater = Seater(pn, limits()).to(self.dev)
        self.max_dur = OFFSETS[-1] + K / min_fps + 0.6
        self.N = int(math.ceil(self.max_dur * SRC_FPS)) + 1  # source frames per crop
        self.Tmax = int(math.ceil(self.max_dur * max_fps)) + 2  # detector frames, padded
        rows = [(k, w) for k, w in clip_table(split)]
        mps = {k: np.load(FITS / f"{k}.npz")["mp"] for k, _ in rows}
        rows = [(k, w) for k, w in rows if len(mps[k]) >= self.N]
        self.keys = [k for k, _ in rows]
        w = torch.tensor([w for _, w in rows], dtype=torch.float32)
        self.p = (w / w.sum()).to(self.dev)
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
        dev, K = self.dev, self.K
        fps = torch.where(torch.rand(B, generator=self.gen, device=dev) < 0.5, torch.full((B,), 10.0, device=dev),
                          self._u(B, self.min_fps, self.max_fps))
        dur = OFFSETS[-1] + K / fps + 0.6
        clip = torch.multinomial(self.p, B, replacement=True, generator=self.gen)
        s0 = torch.floor(torch.rand(B, generator=self.gen, device=dev) * (self.lens[clip] - self.N + 1)).long()
        rows = self.start[clip, None] + s0[:, None] + torch.arange(self.N, device=dev)[None]
        seated = torch.rand(B, generator=self.gen, device=dev) < self.p_seated
        mp = self.seater(self.gen, self.mp[rows], seated, SRC_FPS)

        times, valid = self.frame_times(fps, dur)
        tz = torch.where(valid, times, torch.zeros_like(times))
        src = torch.round(tz * SRC_FPS).long().clamp(max=self.N - 1)
        states = self.fk(mp.gather(1, src[..., None].expand(B, self.Tmax, mp.shape[-1])))  # (B, T, J, 8)
        X = self.lm(states) / 100.0
        style = Style.sample(B, self.gen, dev)
        X = style.placement(X)
        Xn = torch.where(valid[..., None, None], X, torch.full_like(X, math.nan))
        R, t, fx = sample_desk_camera(Xn, self.gen)
        image = project(X, R, t, fx)
        flat = style.flatten(world_landmarks(X, R, t), shoulder_distance(Xn, R, t))
        image[..., 2] = flat_image_z(X, R, t, fx, flat)
        world = style.world(flat)
        dt = torch.diff(tz, dim=1, prepend=(tz[:, :1] - 1.0 / fps[:, None]))
        dt = torch.where(valid, dt, (1.0 / fps)[:, None].expand_as(dt))
        W, Im, vis, present = observe(self.gen, world, image, dt, sample_config(B, self.gen, dev))
        Im[..., 2] = Im[..., 2] * style.image_z[:, None, None]
        present = present & valid

        n_valid = valid.sum(1)
        tgt = (n_valid[:, None] - K + torch.arange(K, device=dev)[None]).clamp_min(0)
        st_t = states.gather(1, tgt[..., None, None].expand(B, K, *states.shape[2:]))
        rot, pos, off = self.targets(st_t, R)
        feats = GF.frame_features(times, W, Im, vis, present, valid)
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
        return dict(x=x, rot=rot, pos=pos, off=off, dt=dt_out, seated=seated.float())
