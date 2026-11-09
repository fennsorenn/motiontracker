"""Training samples generated on the fly: fitted CMU crop -> (seated) -> synthetic observations.

Each sample: K consecutive target frames with their full history (features.OFFSETS), so the velocity
loss can compare neighbouring predictions. Every sample draws a fresh camera, bias style and noise.
"""

import json
import os

import numpy as np
import torch

from mtcore.landmarks import LandmarkModel
from mtcore.mhr import limits, load, skeleton_state
from mtcore.paths import DATA, REPO_ROOT

from .features import OFFSETS, frame_features, history_indices, model_input
from .seated import clamp, seat
from .skeleton import TargetExtractor
from .synth import synthesize

FITS = DATA / "m1" / os.environ.get("MT_FITS", "fits")  # MT_FITS: another fits directory under data/m1
SCAN = REPO_ROOT / "spikes" / "m3-mhr-vrm" / "cmu_scan.json"
SRC_FPS = 30.0
CHOREO = DATA / "m1" / "choreo"  # designed contact choreography pool (mttrain.choreo)
# whole subjects held out for validation (~10% of clips; chosen once, kept fixed)
VAL_SUBJECTS = {13, 14, 26, 61, 86, 104, 118, 143}


def clip_table(split):
    """[(key, weight)] for 'train' or 'val'. Drops clips the fit QC flagged (poor arm fit, CMU glitches)."""
    index = json.loads((FITS / "index.json").read_text())
    scan = {r["clip"]: r for r in json.loads(SCAN.read_text()) if "error" not in r}
    hm = np.median([r["hand_motion"] for r in scan.values()])
    rows = []
    for key, r in index.items():
        if r["arm_res_p95"] > 3.0 or r["max_wrist_speed"] > 10.0 or r["frames"] < 90:
            continue
        if (int(key.split("_")[0]) in VAL_SUBJECTS) != (split == "val"):
            continue
        # weight toward upper-body activity, and toward longer clips (more distinct crops)
        w = np.clip(scan[key]["hand_motion"] / hm, 0.5, 3.0) * np.sqrt(r["frames"] / 300)
        rows.append((key, float(w)))
    return rows


def choreo_table(split):
    """Choreography pool files of a split (mttrain.choreo), [] if none were generated."""
    return sorted((CHOREO / split).glob("c*.npz"))


def source_table(split, p_choreo):
    """Training sources: fitted clips (weights as clip_table) plus the choreography pool, which gets
    `p_choreo` of the total weight. -> [(key, motion (N, P), probability, is_choreo)]."""
    rows = clip_table(split)
    w = np.array([w for _, w in rows])
    out = [(k, np.load(FITS / f"{k}.npz")["mp"], p, False) for (k, _), p in zip(rows, w / w.sum())]
    pool = choreo_table(split) if p_choreo > 0 else []
    if not pool:
        return out
    out = [(k, mp, p * (1 - p_choreo), c) for k, mp, p, c in out]
    return out + [(f"choreo/{f.stem}", np.load(f)["mp"], p_choreo / len(pool), True) for f in pool]


class Generator:
    def __init__(self, split, K=8, p_seated=0.75, seed=0, p_choreo=0.2):
        self.K, self.p_seated = K, p_seated
        self.rng = np.random.default_rng(seed)
        self.ch = load()
        self.pn = list(self.ch.parameter_transform.names)
        jn = list(self.ch.skeleton.joint_names)
        rest = skeleton_state(self.ch, np.zeros(len(self.pn)))
        self.lm = LandmarkModel(self.ch, rest)
        self.te = TargetExtractor(jn, rest)
        self.lim = limits()
        rows = source_table(split, p_choreo)
        self.keys = [k for k, _, _, _ in rows]
        self.p = np.array([p for _, _, p, _ in rows])
        self.mp = {k: mp for k, mp, _, _ in rows}
        self.choreo = {k for k, _, _, c in rows if c}

    def sample(self):
        rng = self.rng
        while True:
            fps = 10.0 if rng.random() < 0.5 else float(rng.uniform(8, 30))
            dur = OFFSETS[-1] + self.K / fps + 0.6
            key = self.keys[rng.choice(len(self.keys), p=self.p)]
            mp = self.mp[key]
            n = int(np.ceil(dur * SRC_FPS)) + 1
            if len(mp) < n:
                continue
            s0 = rng.integers(0, len(mp) - n + 1)
            crop = mp[s0:s0 + n]
            seated = rng.random() < self.p_seated
            if key in self.choreo:  # already seated and clamped
                seated = True
            else:
                crop = seat(rng, self.pn, crop, SRC_FPS, self.lim) if seated else clamp(self.pn, crop, self.lim)
            s = synthesize(rng, self.ch, self.lm, crop, SRC_FPS, fps=fps)
            T = len(s["t"])
            if T < self.K + 2:
                continue
            tgt = np.arange(T - self.K, T)
            rot, pos, off = self.te(s["states"][tgt], s["camera"].R)
            cof = self.te.contact_offsets(s["states"][tgt], s["camera"].R, self.lm(s["states"][tgt]))
            times = s["t"]
            feats = frame_features(times, s["world"], s["image"], s["vis"], s["present"])
            idx, age = history_indices(times, tgt)
            if rng.random() < 0.1:  # session start: no history before a random point
                cut = rng.uniform(times[0], times[tgt[0]])
                idx = np.where(times[np.maximum(idx, 0)] < cut, -1, idx)
            x = model_input(feats, idx, age)
            return dict(x=x, rot=rot.astype(np.float32), pos=pos.astype(np.float32), off=off.astype(np.float32),
                        cof=cof.astype(np.float32),
                        dt=np.diff(times[tgt], prepend=times[tgt[0] - 1]).astype(np.float32),
                        seated=np.float32(seated))


class SampleStream(torch.utils.data.IterableDataset):
    def __init__(self, split="train", K=8, seed=0, p_choreo=0.2):
        self.split, self.K, self.seed, self.p_choreo = split, K, seed, p_choreo

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        gen = Generator(self.split, self.K, seed=self.seed * 1000 + wid, p_choreo=self.p_choreo)
        while True:
            yield {k: torch.from_numpy(np.asarray(v)) for k, v in gen.sample().items()}


def fixed_set(split, n, K=8, seed=12345, p_choreo=0.2):
    """A deterministic batch of n samples (validation)."""
    gen = Generator(split, K, seed=seed, p_choreo=p_choreo)
    samples = [gen.sample() for _ in range(n)]
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}
