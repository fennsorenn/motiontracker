"""Training batches for the merger (plan `dev-notes/plans/m4-merger.md`), built on the GPU per batch.

Per sample: a crop of full MHR parameters at 30 fps from one of two sources,
- the body model's sources (`gpugen` crops: CMU seated clips, choreography, head moves; seater idle life
  low-passed, `smooth_life`) with procedural fingers from a finger bank (`hands.motion`, no tremor), or
- the merger pool (`hands.pool2 --full --no-tremor`: hands in use, rests, body contact, two-hand events) with
  head life and head moves added (`life`);
then detector frames at 10 Hz (the hand model's rate) observed three ways from one camera:
- body: the body model's observation path (`GpuGenerator.observe`) -> frozen `m1` at the last K frames;
- hands: `observe2.PairObserver` on the same camera -> frozen `p3` at the same frames (HISTORY frames each);
- head: the face-mesh head pose (`mtcore.facepose`) simulated from the true head with the error model calibrated
  on renders (`facecal`: per-axis slope, per-sequence offset, within-sequence AR(1), lost runs);
plus real recorded blendshapes, unpaired (owner, 2026-11-14), with dropout.

All rotations are in the body model's frame: camera axes, rest removed (`gpugen.geom.Targets`). In it the hand
model's palm frame F (rows, camera frame) gives the wrist directly: G_wrist = F^T F_rest; the face pose R_face
gives the head: G_head = R_face offset^T rest_head^T.

Returned per sample, for the last K detector frames (oldest first): the part outputs (merger inputs), the base pose
(the hand-written combination, `combine`), and the targets (truth).
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from mtcore.hand_features import palm_frame
from mtcore.mhr import positions, skeleton_state
from mtcore.paths import DATA

from ..gpugen.fk import MhrFK
from ..gpugen.generator import GpuGenerator
from ..gpugen.geom import quat_to_mat
from ..gpugen.motion import head_moves, slow_signals, smooth
from ..hands.model import OUT_IDX
from ..hands.model2 import HISTORY, PairNet, scenes
from ..hands.motion import HandMotion
from ..hands.observe2 import PairObserver
from ..hands.skeleton import HandFK
from ..model import Lifter
from ..seated import HEAD_LIFE, HEAD_LIFE_TAU, HEAD_MOVES, HEAD_SHARES
from ..skeleton import OUT_JOINTS, PARENT_IDX

MERGER = DATA / "merger"
FPS = 10.0  # detector rate (the hand model is trained at 10 Hz)
K = 11  # target frames per sample: merger history offsets up to 8 frames back, at the newest three frames
# (velocity and acceleration losses)
P_POOL = 0.5  # share of samples from the merger pool (hand events) vs the body model's sources
SRC_FPS = 30.0
MIRROR = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
J = {n: i for i, n in enumerate(OUT_JOINTS)}
BLEND_FILES = sorted((DATA / "videos").glob("*-full/face.landmarks.jsonl"))
CONF_DEG = (15.0, 50.0)  # hand model rotation scale -> weight of its wrist (as `hands.wrist`)
MAX_BEND_DEG = 100.0
ARKIT = None  # blendshape names, filled from the first bank file


# --- rotations ----------------------------------------------------------------------------------------------
def geodesic(Ra, Rb):
    c = ((Ra * Rb).sum((-1, -2)) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def euler_yxz(R):
    """camera <- face rotations (..., 3, 3) -> (turn, nod, tilt) radians, R = Ry(a) Rx(b) Rz(c)."""
    b = torch.asin((-R[..., 1, 2]).clamp(-1, 1))
    a = torch.atan2(R[..., 0, 2], R[..., 2, 2])
    c = torch.atan2(R[..., 1, 0], R[..., 1, 1])
    return torch.stack([a, b, c], -1)


def from_euler_yxz(e):
    a, b, c = e.unbind(-1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    o, z = torch.ones_like(a), torch.zeros_like(a)
    Ry = torch.stack([ca, z, sa, z, o, z, -sa, z, ca], -1).unflatten(-1, (3, 3))
    Rx = torch.stack([o, z, z, z, cb, -sb, z, sb, cb], -1).unflatten(-1, (3, 3))
    Rz = torch.stack([cc, -sc, z, sc, cc, z, z, z, o], -1).unflatten(-1, (3, 3))
    return Ry @ Rx @ Rz


def slerp_mat(A, B, w):
    """Rotation from A (w = 0) to B (w = 1), (..., 3, 3), w (...)."""
    D = A.transpose(-1, -2) @ B
    ang = geodesic(D, torch.eye(3, device=D.device).expand_as(D))
    axis = torch.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]], -1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    th = (w * ang)[..., None, None]
    Kx = torch.zeros_like(D)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    Kx = Kx - Kx.transpose(-1, -2)
    Rw = torch.eye(3, device=D.device) + th.sin() * Kx + (1 - th.cos()) * (Kx @ Kx)
    return A @ Rw


def globals_from_locals(L):
    """Body model locals (..., 17, 3, 3) -> globals (rest removed, camera frame)."""
    G = [None] * len(OUT_JOINTS)
    for k, p in enumerate(PARENT_IDX):
        G[k] = L[..., k, :, :] if p < 0 else G[p] @ L[..., k, :, :]
    return torch.stack(G, -3)


def locals_from_globals(G):
    L = torch.empty_like(G)
    for k, p in enumerate(PARENT_IDX):
        L[..., k, :, :] = G[..., k, :, :] if p < 0 else G[..., p, :, :].transpose(-1, -2) @ G[..., k, :, :]
    return L


def unmirror_frame(F):
    """Palm frame rows of mirrored points (right hands in the scene convention) -> frame of the unmirrored points."""
    G = F @ MIRROR.to(F)
    return torch.cat([G[..., :2, :], -G[..., 2:, :]], -2)


def conf_weight(log_s):
    deg = torch.rad2deg(torch.exp(log_s))
    return ((CONF_DEG[1] - deg) / (CONF_DEG[1] - CONF_DEG[0])).clamp(0, 1)


def combine(body_L, wrist_G, wrist_log_s, head_G=None, head_w=None):
    """Hand-written combination (base pose, `hands.wrist.combine` in the body model's frame): body locals
    (..., 17, 3, 3), hand model wrists {side: (..., 3, 3) global}, their log scales {side: (...)}; optional face
    head global (..., 3, 3) with weight (...). Wrists: blended from the body's by the hand model's confidence,
    implausible bends against the body's forearm (> MAX_BEND_DEG off its twist) keep the body's. -> locals."""
    G = globals_from_locals(body_L)
    G = G.clone()
    for s in "lr":
        tw = G[..., J[f"{s}_wrist_twist"], :, :]
        Gw = slerp_mat(G[..., J[f"{s}_wrist"], :, :], wrist_G[s], conf_weight(wrist_log_s[s]))
        local = tw.transpose(-1, -2) @ Gw
        # bend: the part of the local rotation off the forearm axis (the twist joint's x axis in MHR's rest-removed
        # frame is not known here, so the whole local angle is used: a conservative stand-in)
        ok = torch.rad2deg(geodesic(local, torch.eye(3, device=local.device).expand_as(local))) <= MAX_BEND_DEG
        G[..., J[f"{s}_wrist"], :, :] = torch.where(ok[..., None, None], Gw, G[..., J[f"{s}_wrist"], :, :])
    if head_G is not None:
        G[..., J["c_head"], :, :] = slerp_mat(G[..., J["c_head"], :, :], head_G, head_w)
    return locals_from_globals(G)


# --- sources ------------------------------------------------------------------------------------------------
def life(gen, mp, P, fps=SRC_FPS):
    """Head life and head moves (as the seater's, low-passed) on pool sequences (B, N, P); arms untouched."""
    B, n, _ = mp.shape
    dev = mp.device
    S = slow_signals(gen, B, n, fps, list(HEAD_LIFE.values()), [HEAD_LIFE_TAU] * len(HEAD_LIFE), dev)
    S = smooth(S.reshape(-1, n), fps, 0.3).reshape(S.shape)
    out = mp.clone()
    act = torch.rand(B, 1, generator=gen, device=dev)
    for j, key in enumerate(HEAD_LIFE):
        out[:, :, P[key]] += act * S[:, j]
    on = (torch.rand(B, generator=gen, device=dev) < HEAD_MOVES["p"]).float()[:, None, None]
    hm = on * head_moves(gen, B, n, fps, dev)
    for k, (neck, head, share) in enumerate(HEAD_SHARES):
        out[:, :, P[neck]] += share * hm[..., k]
        out[:, :, P[head]] += (1 - share) * hm[..., k]
    return out


def finger_bank(split, n, seconds, seed):
    """Procedural finger sequences (n, T, 2, 27) at 30 fps, no tremor; cached in data/merger/."""
    path = MERGER / f"fingers_{split}.npz"
    if path.exists():
        return np.load(path)["q"]
    rng = np.random.default_rng(seed)
    hm = HandMotion(tremor=False)
    q = np.stack([np.stack([hm.sequence(rng, seconds, SRC_FPS)[0] for _ in "lr"], 1) for _ in range(n)]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, q=q)
    tmp.replace(path)
    return q


def blend_bank():
    """Real recorded ARKit blendshapes at the recorder's 10 Hz: list of (T, 52) arrays, and the names."""
    global ARKIT
    seqs = []
    for f in BLEND_FILES:
        rows = [json.loads(line) for line in open(f) if line.strip()]
        rows = [r.get("blendshapes") for r in rows]
        names = ARKIT or sorted(next(r for r in rows if r))
        ARKIT = names
        seqs.append(np.array([[r.get(k, 0.0) for k in names] if r else [np.nan] * len(names) for r in rows], np.float32))
    return seqs


# --- batches ------------------------------------------------------------------------------------------------
class MergerBatches:
    def __init__(self, split, device, seed=0, m1=DATA / "m1" / "runs" / "m1-v10" / "best.pt",
                 p3=DATA / "m4" / "runs" / "p3" / "best.pt", hand_calib=DATA / "m4" / "calib" / "hands1.json",
                 scores=DATA / "m4" / "calib" / "scores.json", face_calib=MERGER / "calib" / "face.json",
                 pool=MERGER / "pool"):
        self.dev = torch.device(device)
        self.body = GpuGenerator(split, device, K=K, seed=seed, min_fps=FPS, max_fps=FPS)
        self.gen = self.body.gen
        ch = self.body_ch = __import__("mtcore.mhr", fromlist=["load"]).load()
        pn = list(ch.parameter_transform.names)
        jn = list(ch.skeleton.joint_names)
        self.P = {n: i for i, n in enumerate(pn)}
        hfk = HandFK(ch)
        self.fing = {s: torch.tensor(hfk.idx[s], device=self.dev) for s in "lr"}
        self.mp21 = {s: torch.tensor(hfk.mp21[s], device=self.dev) for s in "lr"}
        self.head = jn.index("c_head")
        keep = sorted(set(hfk.mp21["l"].tolist()) | set(hfk.mp21["r"].tolist()) | {self.head})
        self.fk = MhrFK(ch, keep).to(self.dev)
        rest = skeleton_state(ch, np.zeros(len(pn)))
        P0 = positions(ch, np.zeros((1, len(pn)), np.float32))[0]
        self.F_rest = {s: torch.tensor(palm_frame(P0[hfk.mp21[s]])[1], dtype=torch.float32, device=self.dev) for s in "lr"}
        self.rest_head = quat_to_mat(torch.tensor(rest[self.head, 3:7], dtype=torch.float32)).to(self.dev)
        # sources
        z = np.load(pool / f"{split}.npz")
        self.pool = torch.tensor(z["mp"], device=self.dev)  # (N, T, P) at 30 fps
        self.fingers = torch.tensor(finger_bank(split, 400 if split == "train" else 60, 20.0, 7 + (split != "train")),
                                    device=self.dev)
        bl = blend_bank()
        self.blend = [torch.tensor(b, device=self.dev) for b in bl]
        # observers
        self.hands_obs = PairObserver(str(hand_calib), self.dev, str(scores))
        fc = json.load(open(face_calib))
        self.face_offset = torch.tensor(fc["offset"], dtype=torch.float32, device=self.dev)
        self.face = fc["sets"]["all"]
        # frozen part models
        ck = torch.load(m1, map_location="cpu", weights_only=False)
        self.m1 = Lifter().to(self.dev).eval()
        self.m1.load_state_dict(ck["model"])
        sd = torch.load(p3, map_location="cpu")["model"]
        self.p3 = PairNet(frame_dim=sd["frame.0.weight"].shape[1]).to(self.dev).eval()
        self.p3.load_state_dict(sd)
        for p in list(self.m1.parameters()) + list(self.p3.parameters()):
            p.requires_grad_(False)

    def _u(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.dev)

    def _n(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.dev)

    def sources(self, B):
        """Full MHR parameter crops (B, N, P) at 30 fps, and source (B,) 1 = merger pool."""
        g = self.body
        n_pool = int(round(B * P_POOL))
        _, mp_b, _ = g.crop(B - n_pool, fps=FPS, smooth_life=True)
        Nf, Tf = self.fingers.shape[:2]
        k = torch.randint(0, Nf, (B - n_pool,), generator=self.gen, device=self.dev)
        s0 = torch.randint(0, Tf - g.N + 1, (B - n_pool,), generator=self.gen, device=self.dev)
        q = self.fingers[k[:, None], s0[:, None] + torch.arange(g.N, device=self.dev)[None]]  # (b, N, 2, 27)
        for i, s in enumerate("lr"):
            mp_b[:, :, self.fing[s]] = q[:, :, i]
        Np, Tp = self.pool.shape[:2]
        k = torch.randint(0, Np, (n_pool,), generator=self.gen, device=self.dev)
        s0 = torch.randint(0, Tp - g.N + 1, (n_pool,), generator=self.gen, device=self.dev)
        mp_p = life(self.gen, self.pool[k[:, None], s0[:, None] + torch.arange(g.N, device=self.dev)[None]], self.P)
        mp = torch.cat([mp_b, mp_p], 0)
        src = torch.cat([torch.zeros(B - n_pool), torch.ones(n_pool)]).to(self.dev)
        return mp, src

    def head_obs(self, G_head, head_pos, R, t, fx, frames_valid):
        """True head globals (B, T, 3, 3) (rest removed, camera frame), head joint positions (B, T, 3) metres,
        camera -> simulated face-mesh head global (B, T, 3, 3), centre (B, T, 2), log scale (B, T), seen (B, T)."""
        B, T = G_head.shape[:2]
        f = self.face
        R_true = G_head @ self.rest_head @ self.face_offset  # camera <- face model
        e = euler_yxz(R_true)
        obs = []
        for i, ax in enumerate(("turn", "nod", "tilt")):
            c = f[ax]
            slope = c["slope"] + 0.1 * self._n(B, 1)
            off = math.radians(c["bias"]) + math.radians(c["sd_between"]) * self._n(B, 1)
            rho = min(max(c["res_lag1"], 0.0), 0.95)
            eps = self._n(B, T) * math.radians(c["sd_within"]) * math.sqrt(1 - rho ** 2)
            ar = torch.empty_like(eps)
            ar[:, 0] = self._n(B) * math.radians(c["sd_within"])
            for k in range(1, T):
                ar[:, k] = rho * ar[:, k - 1] + eps[:, k]
            obs.append(slope * e[..., i] + off + ar)
        R_obs = from_euler_yxz(torch.stack(obs, -1))
        G_obs = R_obs @ self.face_offset.transpose(-1, -2) @ self.rest_head.transpose(-1, -2)
        C = torch.einsum("bij,btj->bti", R, head_pos - t[:, None])
        z = C[..., 2].clamp_min(0.05)
        centre = fx[:, None, None] * C[..., :2] / z[..., None] + torch.tensor(f["centre_bias"], device=self.dev) \
            + torch.tensor(f["centre_std"], device=self.dev) * self._n(B, T, 2)
        log_scale = torch.log(f["scale_gain"] * fx[:, None] / z) + f["scale_log_std"] * self._n(B, T)
        # lost: runs per sequence (renders: ~19% of frames without a face), always past ~75 deg of turn
        rate = self._u(B, 1, hi=0.25) / 4
        start = self._u(B, T) < rate
        length = torch.randint(1, 6, (B, T), generator=self.gen, device=self.dev)
        lost = torch.zeros(B, T, dtype=torch.bool, device=self.dev)
        for k in range(5):
            sh = torch.roll(start & (length > k), k, 1)
            sh[:, :k] = False
            lost |= sh
        lost |= (e[..., 0].abs() > math.radians(75)) | ~frames_valid
        return G_obs, centre, log_scale, ~lost

    def blendshapes(self, B, T):
        """Real blendshape windows (B, T, 52) and present (B, T); 30% of samples without (dropout)."""
        out = torch.zeros(B, T, len(ARKIT), device=self.dev)
        for b in range(B):
            s = self.blend[int(torch.randint(len(self.blend), (1,), generator=self.gen, device=self.dev))]
            k = int(torch.randint(len(s) - T, (1,), generator=self.gen, device=self.dev))
            out[b] = s[k:k + T]
        drop = self._u(B, 1) < 0.3
        present = torch.isfinite(out).all(-1) & ~drop
        return torch.where(present[..., None], out.nan_to_num(), torch.zeros_like(out)), present

    def jitter(self, R, deg):
        """Small random rotations (sd `deg` per axis) applied to rotations (..., 3, 3): the merger's input jitter
        (owner, 2026-11-14: jitter on the input, none on the output)."""
        if deg <= 0:
            return R
        v = torch.randn(R.shape[:-2] + (3,), generator=self.gen, device=self.dev) * math.radians(deg)
        a = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        k = v / a
        Kx = torch.zeros(R.shape, device=self.dev)
        Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
        Kx = Kx - Kx.transpose(-1, -2)
        a = a[..., None]
        return R @ (torch.eye(3, device=self.dev) + a.sin() * Kx + (1 - a.cos()) * (Kx @ Kx))

    @torch.no_grad()
    def sample(self, B, jitter_deg=0.0):
        """`jitter_deg`: extra input jitter on the part rotations (body locals, hand wrists, face head), before the
        base combination, so base and inputs agree."""
        dev = self.dev
        mp, src = self.sources(B)
        fps = torch.full((B,), FPS, device=dev)
        ob = self.body.observe(mp, fps, drop_rate=0.0)
        tgt, R, t, fx = ob["tgt"], ob["R"], ob["t"], ob["fx"]
        Tv = int(ob["valid"].sum(1).max())
        times = torch.where(ob["valid"], ob["times"], torch.zeros_like(ob["times"]))[:, :Tv]
        idx = torch.round(times * SRC_FPS).long().clamp(max=mp.shape[1] - 1)
        mpd = mp.gather(1, idx[..., None].expand(B, Tv, mp.shape[-1]))  # parameters at the detector frames
        st = self.fk(mpd.reshape(B * Tv, -1)).reshape(B, Tv, -1, 8)
        hands = torch.stack([st[:, :, self.mp21[s], :3] for s in "lr"], 2) / 100.0  # (B, Tv, 2, 21, 3)
        body_lm = self.body.lm(ob["states"][:, :Tv]) / 100.0
        obs, truth = self.hands_obs(hands, body_lm, self.gen, camera=(R, t, fx))

        # body model at the K target frames
        rot, log_s = self.m1(ob["x"])  # (B, K, 17, 3, 3), (B, K, 17)
        rot = self.jitter(rot, jitter_deg)
        # hand model: a HISTORY window ending at each target frame (newest first)
        win = tgt[:, :, None] - torch.arange(HISTORY, device=dev)[None, None]  # (B, K, H)
        g = lambda x: x.gather(1, win.reshape(B, -1).reshape(B, K * HISTORY, *([1] * (x.dim() - 2))).expand(
            B, K * HISTORY, *x.shape[2:])).reshape(B * K, HISTORY, *x.shape[2:]) if x.dim() > 2 else \
            x.gather(1, win.reshape(B, -1)).reshape(B * K, HISTORY)
        own, other = scenes(g(obs["img"]), g(obs["canon"]), g(obs["lost"]), g(obs["score"]))
        out = self.p3(own, other)  # rows: B*K own = left, then B*K own = right (mirrored)
        BK = B * K
        hand = {}
        for i, s in enumerate("lr"):
            sl = slice(i * BK, (i + 1) * BK)
            F = out["rot"][sl] if s == "l" else unmirror_frame(out["rot"][sl])
            hand[s] = {"G": self.jitter((F.transpose(-1, -2) @ self.F_rest[s]).reshape(B, K, 3, 3), jitter_deg),
                       "rot_log_s": out["rot_log_s"][sl].reshape(B, K),
                       "fingers": out["fingers"][sl].reshape(B, K, -1),
                       "fingers_log_s": out["fingers_log_s"][sl].reshape(B, K, -1),
                       "pos": out["pos"][sl].reshape(B, K, 2), "log_size": out["log_size"][sl].reshape(B, K),
                       "lost": obs["lost"][:, :, i].gather(1, tgt), "score": obs["score"][:, :, i].gather(1, tgt)}
        # head from the face mesh, at the target frames
        Gt = quat_to_mat(st[:, :, self.head, 3:7])
        G_head_true = R[:, None] @ Gt @ self.rest_head.T
        hG, hc, hs, hseen = self.head_obs(G_head_true, st[:, :, self.head, :3] / 100.0, R, t, fx, ob["valid"][:, :Tv])
        pick = lambda x: x.gather(1, tgt.reshape(B, K, *([1] * (x.dim() - 2))).expand(B, K, *x.shape[2:]))
        head = {"G": self.jitter(pick(hG), jitter_deg), "centre": pick(hc), "log_scale": pick(hs), "seen": pick(hseen)}
        bs, bs_present = self.blendshapes(B, K)

        base = combine(rot, {s: hand[s]["G"] for s in "lr"}, {s: hand[s]["rot_log_s"] for s in "lr"},
                       head["G"], head["seen"].float() * 0.8)
        true_fingers = torch.stack([pick(mpd)[..., self.fing[s]] for s in "lr"], 2)  # (B, K, 2, 27)
        return {"body": {"rot": rot, "log_s": log_s}, "hand": hand, "head": head,
                "blend": {"values": bs, "present": bs_present}, "base": base, "src": src,
                "target": {"rot": ob["rot"], "pos": ob["pos"], "off": ob["off"], "cof": ob["cof"], "fingers": true_fingers,
                           "seated": torch.ones(B, device=dev),
                           "inside": truth["inside"].gather(1, tgt[..., None].expand(B, K, 2))}}
