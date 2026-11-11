"""Upper bound for retargeting at contacts: plain rotation copy vs IK onto the avatar's own body.

    cd train && uv run python ../spikes/retarget-upper-bound/run.py [--step 3] [--lam 4]

Input: the rendered contact set contact1 (true MHR poses `seq.npz` mp + skinned verts, contact holds
in `contacts.json`). For each held-out performer body and each VRoid sample avatar:

- copy: MHR -> VRM normalized rotations (`mtcore.convert.MhrToVrm`, built on the performer's own
  rest proportions) and avatar FK. This is what the tracker output gets today.
- ik: during each hold, the copied arm rotations (upper arm, forearm, hand; both arms for palms)
  are adjusted so the avatar's hand point lands on the corresponding target point of the avatar's
  own body, with the performer's offset (contact ~1 cm, near miss 8-14 cm, scaled by arm length),
  staying as close to the copy as possible (penalty lam * |delta rad|^2, cm^2). The ideal input a
  retargeting layer could get: the true pose and the true target.

Correspondence (performer mesh -> avatar mesh): per frame the performer's closest hand / target
vertex pair; each point is expressed in its body part's frame (head: eye midpoint, left-right
eyes, up from the neck; hand: wrist, toward the middle knuckle, index -> pinky knuckle), normalized
by the part's rest bounding box, mapped into the avatar's part frame and snapped to the avatar's
nearest surface vertex. Measures on the meshes during the holds: surface gap (closest distance
between the hand and the target part; unsigned, so penetration reads as touching), location error
(how far the touch lands from the corresponding point), arm change of the IK solution, and for
near misses the gap relative to the performer's gap scaled by arm length.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from mtcore.convert import MhrToVrm, vrm_fk
from mtcore.mhr import load, skeleton_state
from mtcore.paths import AVATARS, DATA

sys.path.insert(0, str(Path(__file__).parent))
from avatar_mesh import load_avatar  # noqa: E402

AVATAR_NAMES = ["AvatarSample_D", "AvatarSample_D_Darkness", "AvatarSample_E", "AvatarSample_F", "AvatarSample_G",
                "Base_Female", "Base_Male", "HairSample_Female", "HairSample_Male", "Sakurada_Fumiriya", "Sendagaya_Shino"]
SET = DATA / "m1" / "renders" / "contact1"
OUT = DATA / "spikes" / "retarget"
SIDES = {"left": "l", "right": "r"}


def qmat(q):
    x, y, z, w = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame(o, xdir, ydir):
    x = unit(xdir)
    y = unit(ydir - (ydir * x).sum(-1, keepdims=True) * x)
    return o, np.stack([x, y, np.cross(x, y)], -1)  # columns = axes


# part frames from named points: (origin, x direction, y direction)
def head_pts(get):
    e = 0.5 * (get("leye") + get("reye"))
    return e, get("leye") - get("reye"), e - get("neck")


def hand_pts(get):
    return get("wrist"), get("middle1") - get("wrist"), get("index1") - get("pinky1")


MHR_NAMES = {"leye": "l_eye", "reye": "r_eye", "neck": "c_neck"}
VRM_NAMES = {"leye": "leftEye", "reye": "rightEye", "neck": "neck"}


def mhr_get(st, jn, side=None):
    m = SIDES.get(side)
    names = {**MHR_NAMES, "wrist": f"{m}_wrist", "middle1": f"{m}_middle1", "index1": f"{m}_index1", "pinky1": f"{m}_pinky1"}
    return lambda k: st[..., jn.index(names[k]), :3]


def vrm_get(P, side=None):
    names = {**VRM_NAMES, "wrist": f"{side}Hand", "middle1": f"{side}MiddleProximal", "index1": f"{side}IndexProximal",
             "pinky1": f"{side}LittleProximal"}
    return lambda k: P[names[k]]


def part_frame(kind, get):
    return frame(*(head_pts(get) if kind == "head" else hand_pts(get)))


def box(V, o, R):
    L = (V - o) @ R
    return 0.5 * (L.min(0) + L.max(0)), L.max(0) - L.min(0)


class Performer:
    def __init__(self, ch, seq_dir):
        d = np.load(seq_dir / "seq.npz")
        self.mp = d["mp"].astype(np.float64)
        self.verts = d["verts"]
        self.faces = np.asarray(ch.mesh.faces, dtype=np.int64)
        self.jn = list(ch.skeleton.joint_names)
        par = list(ch.skeleton.joint_parents)
        pn = list(ch.parameter_transform.names)
        body = np.where([n.startswith("scale_") for n in pn], self.mp[0], 0.0)
        self.rest = skeleton_state(ch, body)
        rest_v = np.asarray(ch.skin_points(self.rest.astype(np.float32)))
        dom = np.asarray(ch.skin_weights.to_dense(len(self.jn))).argmax(1)

        def subtree(name):
            out = {self.jn.index(name)}
            grown = True
            while grown:
                grown = False
                for k, p in enumerate(par):
                    if p in out and k not in out:
                        out.add(k)
                        grown = True
            return np.isin(dom, list(out))
        self.groups = {"head": subtree("c_head"), "leftHand": subtree("l_wrist"), "rightHand": subtree("r_wrist")}
        self.box = {}
        for g, m in self.groups.items():
            side = None if g == "head" else g[:-4]
            o, R = part_frame("head" if g == "head" else "hand", mhr_get(self.rest, self.jn, side))
            self.box[g] = box(rest_v[m], o, R)
        J = lambda n: self.rest[self.jn.index(n), :3]
        self.arm = np.linalg.norm(J("l_lowarm") - J("l_uparm")) + np.linalg.norm(J("l_wrist") - J("l_lowarm"))


class Avatar:
    def __init__(self, name):
        a = load_avatar(AVATARS / f"{name}.vrm")
        self.name, self.rest, self.groups = name, a["rest"], a["groups"]
        self.box = {}
        for g, (V, _, _) in self.groups.items():
            side = None if g == "head" else g[:-4]
            o, R = part_frame("head" if g == "head" else "hand", vrm_get(self.rest, side))
            self.box[g] = box(V, o, R)
        r = self.rest
        self.arm = np.linalg.norm(r["leftLowerArm"] - r["leftUpperArm"]) + np.linalg.norm(r["leftHand"] - r["leftLowerArm"])

    def posed(self, g, P, G):
        V, bones, N = self.groups[g]
        out, nrm = np.empty_like(V), np.empty_like(N)
        for b in np.unique(bones):
            m = bones == b
            R = qmat(G[b])
            out[m] = P[b] + (V[m] - self.rest[b]) @ R.T
            nrm[m] = N[m] @ R.T
        return out, nrm


def vertex_normals(V, faces):
    """Area-weighted vertex normals, V (F, N, 3)."""
    a, b, c = V[:, faces[:, 0]], V[:, faces[:, 1]], V[:, faces[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, (slice(None), faces[:, k]), fn)
    return N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)


def map_point(p, g, perf_frame, perf_box, av_frame, av_box):
    (o, R), (c, e) = perf_frame, perf_box
    u = (((p - o) @ R) - c) / e
    (oa, Ra), (ca, ea) = av_frame, av_box
    return oa + Ra @ (ca + u * ea)


def rodrigues(v):
    th = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=v.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


class Chain:
    """Arm chain of one side over a batch of frames, from the copied pose; points rigid with the hand."""

    def __init__(self, av, side, Ps, Gs, Ls):
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
        s = side
        self.Gsh = T([qmat(G[f"{s}Shoulder"]) for G in Gs])
        self.Pua = T([P[f"{s}UpperArm"] for P in Ps])
        self.q = [T([qmat(L[b]) for L in Ls]) for b in (f"{s}UpperArm", f"{s}LowerArm", f"{s}Hand")]
        r = av.rest
        self.o_la = T(r[f"{s}LowerArm"] - r[f"{s}UpperArm"])
        self.o_h = T(r[f"{s}Hand"] - r[f"{s}LowerArm"])
        self.Ph0 = np.stack([P[f"{s}Hand"] for P in Ps])
        self.Rh0 = np.stack([qmat(G[f"{s}Hand"]) for G in Gs])

    def local(self, V):  # world points at the copied pose (F, N, 3) -> hand-local
        return torch.tensor(np.einsum("fji,fnj->fni", self.Rh0, V - self.Ph0[:, None]), dtype=torch.float64)

    def fk(self, delta):  # delta (F, 3, 3) axis-angle per bone (local)
        Gua = self.Gsh @ self.q[0] @ rodrigues(delta[:, 0])
        Pla = self.Pua + (Gua @ self.o_la[:, None])[..., 0]
        Gla = Gua @ self.q[1] @ rodrigues(delta[:, 1])
        Ph = Pla + (Gla @ self.o_h[:, None])[..., 0]
        Gh = Gla @ self.q[2] @ rodrigues(delta[:, 2])
        return self.Pua, Pla, Ph, Gh

    @staticmethod
    def apply(Ph, Gh, l):  # hand-local points (F, N, 3) -> world
        return Ph[:, None] + torch.einsum("fij,fnj->fni", Gh, l)


def solve(chains, side, A_loc, a_loc, tgt, g, lam, lam_hand, w_pen=10.0, rounds=4, iters=30):
    """IK: the hand point a (local to `side`'s hand) to b + n g on the target surface (touch: g ~ 0.3
    cm; near miss: the scaled gap), no hand vertex A inside the target (planes at the nearest target
    vertices, re-assigned each round; face only), lam |delta|^2 to stay near the copy. tgt: dict(B, N, b, n)
    as world arrays (face, fixed) or local to chain tgt['chain'] (other hand); trees per frame."""
    F = len(g)
    delta = {s: torch.zeros(F, 3, 3, dtype=torch.float64, requires_grad=True) for s in chains}
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)
    gg = T(g)[:, None]

    def tworld(delta_):
        if "chain" not in tgt:
            return lambda x: T(x), lambda n: T(n)
        _, _, Ph, Gh = chains[tgt["chain"]].fk(delta_[tgt["chain"]])
        return (lambda x: Chain.apply(Ph, Gh, T(x)) if T(x).dim() == 3 else Ph + torch.einsum("fij,fj->fi", Gh, T(x)),
                lambda n: torch.einsum("fij,fnj->fni", Gh, T(n)) if T(n).dim() == 3 else torch.einsum("fij,fj->fi", Gh, T(n)))

    def hand(delta_, l):
        _, _, Ph, Gh = chains[side].fk(delta_[side])
        return Chain.apply(Ph, Gh, l)

    for _ in range(rounds):
        with torch.no_grad():
            A = hand(delta, A_loc).numpy()
            if "chain" in tgt:  # into the target hand's local frame
                _, _, Ph, Gh = chains[tgt["chain"]].fk(delta[tgt["chain"]])
                A = np.einsum("fji,fnj->fni", Gh.numpy(), A - Ph.numpy()[:, None])
            K = np.stack([tr.query(x)[1] for tr, x in zip(tgt["trees"], A)])
        Bk = np.take_along_axis(tgt["B"], K[..., None], 1)
        Nk = np.take_along_axis(tgt["N"], K[..., None], 1)
        opt = torch.optim.LBFGS(list(delta.values()), lr=1, max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pw, nw = tworld(delta)
            A_ = hand(delta, A_loc)
            sd = ((A_ - pw(Bk)) * nw(Nk)).sum(-1)
            a_ = hand(delta, a_loc)[:, 0]
            r = a_ - (pw(tgt["b"]) + nw(tgt["n"]) * gg)
            pen = w_pen * (torch.relu(-sd) ** 2).sum() if "chain" not in tgt else 0.0
            reg = sum(lam * (d[:, :2] ** 2).sum() + lam_hand * (d[:, 2] ** 2).sum() for d in delta.values())
            loss = (r ** 2).sum() + pen + reg
            loss.backward()
            return loss
        opt.step(closure)
    with torch.no_grad():
        pw, nw = tworld(delta)
        res = np.linalg.norm((hand(delta, a_loc)[:, 0] - (pw(tgt["b"]) + nw(tgt["n"]) * gg)).numpy(), axis=-1)
    return {s: d.detach() for s, d in delta.items()}, res


def angle(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def gap(A, B, NB):
    """Signed closest distance from point set A to surface B per frame (sign from B's normal at the
    nearest vertex: negative = A reaches inside B, its magnitude ~ the penetration depth), and the
    point of B where that happens."""
    out, near = [], []
    for a, b, nb in zip(A, B, NB):
        d, k = cKDTree(b).query(a)
        sd = d * np.sign(((a - b[k]) * nb[k]).sum(-1))
        i = int(sd.argmin())
        out.append(sd[i])
        near.append(b[k[i]])
    return np.array(out), np.array(near)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3, help="hold frames subsampling")
    ap.add_argument("--lam", type=float, default=4.0, help="IK change penalty, upper arm and forearm, cm^2 per rad^2")
    ap.add_argument("--lam-hand", type=float, default=40.0, help="IK change penalty, hand rotation, cm^2 per rad^2")
    ap.add_argument("--avatars", nargs="*", default=AVATAR_NAMES)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ch = load()
    avatars = [Avatar(n) for n in a.avatars]
    rows = []
    for seq in sorted(SET.glob("seq*")):
        perf = Performer(ch, seq)
        jn = perf.jn
        events = json.loads((seq / "contacts.json").read_text())
        convs = {av.name: MhrToVrm(jn, perf.rest, av.rest) for av in avatars}
        for ev in events:
            f0, f1 = int(round(ev["t0"] / 1000 * 30)), int(round(ev["t1"] / 1000 * 30))
            frames = np.arange(f0, min(f1, len(perf.mp)), a.step)
            st = skeleton_state(ch, perf.mp[frames])
            V = perf.verts[frames] * 100.0
            NV = vertex_normals(V, perf.faces)
            palms = ev["variant"] == "palms-together"
            if palms:
                side, tgt = "left", "rightHand"
            else:  # the hand nearer the face
                hd = perf.groups["head"]
                gl = gap(V[:, perf.groups["leftHand"]], V[:, hd], NV[:, hd])[0].mean()
                gr = gap(V[:, perf.groups["rightHand"]], V[:, hd], NV[:, hd])[0].mean()
                side, tgt = ("left", "head") if gl < gr else ("right", "head")
            hg = f"{side}Hand"
            Hp, Tp = V[:, perf.groups[hg]], V[:, perf.groups[tgt]]
            # performer: closest pair per frame, its offset, both in part frames
            pf_gap, pi, pj = [], [], []
            for h, t in zip(Hp, Tp):
                d, k = cKDTree(t).query(h)
                i = int(d.argmin())
                pf_gap.append(d[i]); pi.append(h[i]); pj.append(t[k[i]])
            pf_gap, pi, pj = np.array(pf_gap), np.array(pi), np.array(pj)
            pf_gap = gap(Hp, Tp, NV[:, perf.groups[tgt]])[0]  # signed, as on the avatars
            tkind, tside = ("head", None) if tgt == "head" else ("hand", "right")
            for av in avatars:
                Ls = [convs[av.name].locals(s) for s in st]
                PG = [vrm_fk(L, av.rest) for L in Ls]
                Ps, Gs = [p for p, _ in PG], [g for _, g in PG]
                Ha = np.stack([av.posed(hg, P, G)[0] for P, G in PG])
                Ta, TNa = (np.stack(x) for x in zip(*[av.posed(tgt, P, G) for P, G in PG]))
                copy_gap, copy_touch = gap(Ha, Ta, TNa)
                s = av.arm / perf.arm
                ia, ja, offs = [], [], []
                for k in range(len(frames)):
                    getp = lambda side_: mhr_get(st[k], jn, side_)
                    pf_h, pf_t = part_frame("hand", getp(side)), part_frame(tkind, getp(tside))
                    av_h, av_t = part_frame("hand", vrm_get(Ps[k], side)), part_frame(tkind, vrm_get(Ps[k], tside))
                    ph = map_point(pi[k], hg, pf_h, perf.box[hg], av_h, av.box[hg])
                    pt = map_point(pj[k], tgt, pf_t, perf.box[tgt], av_t, av.box[tgt])
                    ia.append(int(cKDTree(Ha[k]).query(ph)[1]))
                    ja.append(int(cKDTree(Ta[k]).query(pt)[1]))
                    o_local = (pi[k] - pj[k]) @ pf_t[1]  # performer offset in the target part's frame
                    offs.append(av_t[1] @ o_local * s)
                ia, ja, offs = np.array(ia), np.array(ja), np.array(offs)
                F = np.arange(len(frames))
                chains = {side: Chain(av, side, Ps, Gs, Ls)}
                if palms:
                    chains["right"] = Chain(av, "right", Ps, Gs, Ls)
                a_loc = chains[side].local(Ha[F, ia][:, None])
                A_loc = chains[side].local(Ha[:, ::3])
                TNa_ = TNa
                if palms:
                    rc = chains["right"]
                    B = rc.local(Ta).numpy()
                    N = np.einsum("fji,fnj->fni", rc.Rh0, TNa)
                    tgt_spec = dict(chain="right", B=B, N=N, b=B[F, ja], n=N[F, ja], trees=[cKDTree(x) for x in B])
                else:
                    tgt_spec = dict(B=Ta, N=TNa_, b=Ta[F, ja], n=TNa_[F, ja], trees=[cKDTree(x) for x in Ta])
                g_goal = np.full(len(F), 0.3) if ev["label"] == "contact" else np.maximum(pf_gap, 0.3) * s
                delta, res = solve(chains, side, A_loc, a_loc, tgt_spec, g_goal, a.lam, a.lam_hand)
                with torch.no_grad():
                    posed = {}
                    for sd, c in chains.items():
                        Pua, Pla, Ph, Gh = c.fk(delta[sd])
                        posed[sd] = (Pua.numpy(), Pla.numpy(), Ph.numpy(), Gh.numpy())
                    def world_group(g, Vcopy):
                        sd = g[:-4]
                        if g == "head" or sd not in chains:
                            return Vcopy
                        l = chains[sd].local(Vcopy)
                        return Chain.apply(torch.tensor(posed[sd][2]), torch.tensor(posed[sd][3]), l).numpy()
                    Hi, Ti = world_group(hg, Ha), world_group(tgt, Ta)
                    TNi = TNa
                    if palms:  # the target hand's normals turn with its IK rotation
                        Rr = np.einsum("fij,fkj->fik", posed["right"][3], chains["right"].Rh0)
                        TNi = np.einsum("fij,fnj->fni", Rr, TNa)
                ik_gap, ik_touch = gap(Hi, Ti, TNi)
                # arm change, IK vs copy (active side)
                Pua, Pla, Ph, Gh = posed[side]
                c_la = np.stack([P[f"{side}LowerArm"] for P in Ps]); c_h = np.stack([P[f"{side}Hand"] for P in Ps])
                d_up = angle(Pla - Pua, c_la - Pua)
                d_fore = angle(Ph - Pla, c_h - c_la)
                d_hand = np.degrees(np.arccos(np.clip((np.trace(np.einsum("fji,fjk->fik", chains[side].Rh0, Gh), axis1=1, axis2=2) - 1) / 2, -1, 1)))
                corr = Ta[F, ja] if not palms else Ti[F, ja]
                # the corresponding pair: offset along the target normal (gap) and across it (slide)
                def pair(H, Tt, TN):
                    d = H[F, ia] - Tt[F, ja]
                    n = TN[F, ja]
                    gn = (d * n).sum(-1)
                    return gn, np.linalg.norm(d - gn[:, None] * n, axis=-1)
                cp_gap, cp_slide = pair(Ha, Ta, TNa)
                ik_pgap, ik_slide = pair(Hi, Ti, TNi)
                pp = (pi - pj)
                pn_ = np.stack([NV[k, perf.groups[tgt]][cKDTree(Tp[k]).query(pj[k])[1]] for k in range(len(F))])
                perf_pair = (pp * pn_).sum(-1)
                rows.append(dict(seq=seq.name, label=ev["label"], variant=ev["variant"], side=side, avatar=av.name,
                                 scale=float(s), perf_gap=float(np.median(pf_gap)),
                                 copy_gap=float(np.median(copy_gap)), ik_gap=float(np.median(ik_gap)),
                                 copy_loc=float(np.median(np.linalg.norm(copy_touch - (Ta[F, ja]), axis=-1))),
                                 ik_loc=float(np.median(np.linalg.norm(ik_touch - corr, axis=-1))),
                                 ik_res=float(np.median(res)), perf_pair=float(np.median(perf_pair)),
                                 copy_pair=float(np.median(cp_gap)), copy_slide=float(np.median(cp_slide)),
                                 ik_pair=float(np.median(ik_pgap)), ik_slide=float(np.median(ik_slide)),
                                 d_up=float(np.median(d_up)), d_fore=float(np.median(d_fore)), d_hand=float(np.median(d_hand)),
                                 n=len(frames)))
            print(f"{seq.name} {ev['variant']:15s} {side:5s} performer gap {np.median(pf_gap):4.1f} cm; "
                  + " ".join(f"{r['avatar'][:10]} {r['copy_gap']:4.1f}->{r['ik_gap']:4.1f}" for r in rows[-len(avatars):]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    report(rows)


def report(rows):
    print("\nmedians over events, cm / deg. min gap: closest signed distance hand -> face surface (negative =")
    print("inside the head; face only). pair: the corresponding hand/target points, offset along the target's")
    print("normal (gap) and across it (slide). change: IK vs copy, upper arm / forearm direction, hand rotation.")
    for variant in ("palms-together", "hand-to-face", "near-face"):
        R = [r for r in rows if r["variant"] == variant]
        if not R:
            continue
        m = lambda S, k: np.median([r[k] for r in S])
        na = len({r['avatar'] for r in R})
        print(f"\n== {variant} ({len(R) // na} events x {na} avatars); "
              f"performer: min gap {m(R, 'perf_gap'):.1f}, pair gap {m(R, 'perf_pair'):.1f}")
        print(f"{'':24s} {'--------- copy ---------':>26s}   {'---------- ik ----------':>26s}   {'------ change ------':>21s}")
        print(f"{'avatar':24s} {'min gap':>8s} {'pair gap':>9s} {'slide':>7s}   {'min gap':>8s} {'pair gap':>9s} {'slide':>7s}   "
              f"{'upper':>6s} {'fore':>6s} {'hand':>6s}")
        for name in sorted({r["avatar"] for r in R}) + ["ALL"]:
            S = [r for r in R if name in ("ALL", r["avatar"])]
            mg = (lambda k: f"{m(S, k):8.1f}") if variant != "palms-together" else (lambda k: f"{'-':>8s}")
            print(f"{name:24s} {mg('copy_gap')} {m(S, 'copy_pair'):9.1f} {m(S, 'copy_slide'):7.1f}   {mg('ik_gap')} "
                  f"{m(S, 'ik_pair'):9.1f} {m(S, 'ik_slide'):7.1f}   {m(S, 'd_up'):6.1f} {m(S, 'd_fore'):6.1f} {m(S, 'd_hand'):6.1f}")


if __name__ == "__main__":
    main()
