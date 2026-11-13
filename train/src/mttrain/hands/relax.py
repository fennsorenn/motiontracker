"""Mild collision physics for authored hand motion (owner, 2026-11-12): fingers pushed out of each other
instead of passing through (rubbing, palms together), staying as close to the intended pose as possible.

Per frame, a few dozen gradient steps through MHR's forward kinematics (torch, `gpugen.fk.MhrFK`) on both
hands' finger parameters, plus a little of the wrists: loss = squared capsule overlap between the two arms
(`collide` capsules, made differentiable here) + a pull back to the intended pose; parameters clamped to
MHR's limits. Contact is believable (surfaces touch, fingers flatten or bend away), not soft-tissue accurate.
With `mesh` set (a `collide.BodyMesh` at the sequence's pose), hand joints are also kept out of the body's skin
(owner, 2026-11-13: pushing two hands on the belly apart pushed fingers into the belly): a penalty on joint
centres closer than BODY_CLEAR_CM to the skin, linearised at the closest skin point and its normal (refreshed
every few steps).
"""

import numpy as np
import torch

from mtcore.mhr import limits

from ..gpugen.fk import MhrFK
from .collide import Collider


def seg_dist_t(a1, b1, a2, b2):
    """torch version of collide.seg_dist."""
    d1, d2, r = b1 - a1, b2 - a2, a1 - a2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    den = (a * e - b * b).clamp_min(1e-9)
    s = ((b * f - c * e) / den).clamp(0, 1)
    t = (b * s + f) / e.clamp_min(1e-9)
    t_c = t.clamp(0, 1)
    s = torch.where((t < 0) | (t > 1), ((b * t_c - c) / a.clamp_min(1e-9)).clamp(0, 1), s)
    return (a1 + d1 * s[..., None] - a2 - d2 * t_c[..., None]).norm(dim=-1)


BODY_CLEAR_CM = 0.6


class Relaxer:
    def __init__(self, ch, device="cpu", steps=60, lr=0.03, prior=0.02, wrist_prior=0.5, margin=0.1):
        self.col = Collider(ch)
        pn = list(ch.parameter_transform.names)
        self.pn = pn
        (la, lb, self.lr_), (ra, rb, self.rr_) = self.col.l, self.col.r
        self.idx = {k: torch.tensor(v) for k, v in (("la", la), ("lb", lb), ("ra", ra), ("rb", rb))}
        keep = sorted(set(np.concatenate([la, lb, ra, rb]).tolist()))
        self.fk = MhrFK(ch, keep=keep).to(device)
        fingers = [k for k, n in enumerate(pn) if n[:2] in ("l_", "r_") and any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
        wrists = [k for k, n in enumerate(pn) if n[2:] in ("wrist_ry", "wrist_rz")]
        self.free = torch.tensor(fingers + wrists)
        self.weight = torch.tensor([prior] * len(fingers) + [wrist_prior] * len(wrists))
        lim = limits()
        self.lo = torch.tensor([lim.get(pn[k], (-3.0, 3.0))[0] for k in self.free.tolist()])
        self.hi = torch.tensor([lim.get(pn[k], (-3.0, 3.0))[1] for k in self.free.tolist()])
        self.radii = torch.tensor(self.lr_)[:, None] + torch.tensor(self.rr_)[None, :]
        self.radii[-1, -1] -= 2.0  # forearm-forearm: as collide (thick capsules resting on each other)
        self.steps, self.lrate, self.margin, self.device = steps, lr, margin, device
        jn = list(ch.skeleton.joint_names)
        self.hand_joints = torch.tensor([j for j in keep if not jn[j].endswith(("lowarm", "uparm"))])
        self.mesh = None  # collide.BodyMesh at the current pose: keep the hands out of the body too

    def overlap(self, mp):
        P = self.fk(mp)[..., :3]
        d = seg_dist_t(P[:, self.idx["la"], None], P[:, self.idx["lb"], None], P[:, None, self.idx["ra"]], P[:, None, self.idx["rb"]])
        return self.radii - d  # (B, nL, nR) cm

    def __call__(self, mp):
        """mp (T, P) numpy -> relaxed copy, and the deepest overlap per frame before / after (cm)."""
        x0 = torch.tensor(np.asarray(mp, np.float32))
        with torch.no_grad():
            before = self.overlap(x0).amax((1, 2))
        need = before > self.margin
        out = x0.clone()
        if need.any():
            base = x0[need]
            v0 = base[:, self.free].clone()
            v = v0.clone().requires_grad_(True)
            opt = torch.optim.Adam([v], lr=self.lrate)
            body = self.mesh is not None and self.mesh.tree is not None
            for step in range(self.steps):
                full = base.clone()
                full[:, self.free] = v
                P = self.fk(full)[..., :3]
                d = seg_dist_t(P[:, self.idx["la"], None], P[:, self.idx["lb"], None], P[:, None, self.idx["ra"]], P[:, None, self.idx["rb"]])
                ov = self.radii - d
                loss = (torch.relu(ov + self.margin) ** 2).sum((1, 2)).sum() + ((v - v0) ** 2 * self.weight).sum() * 100
                if body:
                    H = P[:, self.hand_joints]
                    if step % 5 == 0:  # closest skin points and normals, linearised for the next steps
                        c, n, _ = self.mesh.closest(H.detach().numpy().astype(np.float64))
                        c, n = torch.tensor(c, dtype=H.dtype), torch.tensor(n, dtype=H.dtype)
                    loss = loss + (torch.relu(BODY_CLEAR_CM - ((H - c) * n).sum(-1)) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v.copy_(torch.maximum(torch.minimum(v, self.hi), self.lo))
            full = base.clone()
            full[:, self.free] = v.detach()
            out[need] = full
        with torch.no_grad():
            after = self.overlap(out).amax((1, 2))
        return out.numpy().astype(np.asarray(mp).dtype), before.numpy(), after.numpy()
