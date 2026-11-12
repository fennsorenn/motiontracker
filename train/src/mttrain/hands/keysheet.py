"""Contact sheet of interaction key poses for visual review (owner, 2026-11-12: confirm poses before rendering
whole sets).

    uv run python -m mttrain.hands.keysheet --out <dir> [--seed 0] [--body 0] [--events palms,rub,...]

For each event, one key pose through the real pipeline (`pairs.PairChoreo._solve`: contacts against the
actual pose, wrist anatomy, collision relaxation), rendered in Blender as a front close-up of the hands on
the body of held-out clip `--body`; tiles labelled with event, overlap after relaxation, wrist flexion and
deviation per hand, and PASS / FAIL (with the reason). Events that cannot be solved show a FAIL tile.
Writes <dir>/sheet.png and <dir>/labels.json. CPU + GPU (Blender): use tools/guarded-run.sh.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from mtcore.landmarks import LandmarkModel
from mtcore.mhr import limits, load, positions, skeleton_state
from mtcore.paths import REPO_ROOT

from .. import choreo
from ..dataset import FITS, clip_table
from ..seated import seat
from . import pairs
from .pairs import CONTACT, PALM_CHECKED, PairChoreo, wrist_angles

DEFAULT = ["palms", "rub", "clap", "steeple", "tap", "count_on_hand", "approach", "cross", "occlude", "turn", "wrist",
           "count_on_hand"]
BLENDER_SCRIPT = REPO_ROOT / "tools" / "render" / "render_blender.py"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--body", type=int, default=0)
    ap.add_argument("--events", default=",".join(DEFAULT))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ch = load()
    pn = list(ch.parameter_transform.names)
    jn = list(ch.skeleton.joint_names)
    lm = LandmarkModel(ch, skeleton_state(ch, np.zeros(len(pn))))
    lim = limits()
    pc = PairChoreo(ch, lm, pn, lim)
    rng = np.random.default_rng(a.seed)
    key = clip_table("val")[a.body % len(clip_table("val"))][0]
    base = np.load(FITS / f"{key}.npz")["mp"][0].astype(np.float64)
    pc.set_body(np.where([n.startswith("scale_") for n in pn], base, 0.0))
    m = seat(rng, pn, np.repeat(pc.body[None].astype(np.float64), 3, 0), 30.0, lim, force_rest=("lap", "lap"), moves=False)[1]
    pairs.still_head(m[None], pn)
    X = lm(skeleton_state(ch, m)[None])[0]
    sh, lat, up, fwd = choreo._body_frame(X)
    labels, tiles = [], []
    for i, kind in enumerate(a.events.split(",")):
        sol, why = None, "no solve"
        for slack in ((0.0, 1.0, 2.0) if kind in CONTACT else (0.0, 0.0, 0.0)):
            keys = pc.event(rng, kind, X, slack)
            k = int(rng.integers(len(keys)))
            sol = pc._solve(m, keys[k][2], kind in PALM_CHECKED)
            if sol is not None:
                break
        if sol is None:
            labels.append(f"{i}: {kind}: FAIL ({dict(pc.rejects)})")
            pc.rejects.clear()
            tiles.append(None)
            continue
        pc.rejects.clear()
        mm = m.copy()
        for s, (arm, q) in sol.items():
            mm[pc.arm[s]] = arm
            mm[pc.fing[s]] = q
        mm, before, after = pc.relaxer(mm[None])
        mm = mm[0]
        P = positions(ch, mm[None].astype(np.float32))[0]
        wr = {s: [int(round(x)) for x in wrist_angles(P[jn.index(f"{s}_lowarm")], P[pc.hm.fk.mp21[s]], s)] for s in sol}
        bad = []
        if after[0] > pairs.FINAL_OVERLAP_CM:
            bad.append(f"overlap {after[0]:.1f}")
        if any(not (pairs.WRIST_FLEX[0] <= f <= pairs.WRIST_FLEX[1] and pairs.WRIST_DEV[0] <= d <= pairs.WRIST_DEV[1]) for f, d in wr.values()):
            bad.append("wrist")
        labels.append(f"{i}: {kind} | overlap {before[0]:.1f} -> {after[0]:.1f} cm | wrist flex,dev {wr} | "
                      + ("PASS" if not bad else "FAIL " + ", ".join(bad)))
        verts = np.asarray(ch.skin_points(skeleton_state(ch, mm[None]).astype(np.float32))) / 100.0
        c = np.mean([P[pc.hm.fk.mp21[s]][9] for s in sol], 0) / 100.0
        pos = c + 0.55 * fwd + 0.06 * up
        f_ = c - pos
        f_ /= np.linalg.norm(f_)
        dn = -(up - (up @ f_) * f_)
        dn /= np.linalg.norm(dn)
        npz = a.out / f"k{i:02d}.npz"
        np.savez_compressed(npz, verts=verts.astype(np.float32), faces=np.asarray(ch.mesh.faces, np.int32),
                            R=np.stack([np.cross(dn, f_), dn, f_]), t=pos, fx=0.5 / np.tan(np.radians(25)),
                            mp=mm[None].astype(np.float32), fps=30.0, skin=np.array([0.78, 0.58, 0.46]),
                            wall=np.array([0.75, 0.78, 0.82]), clip=kind)
        subprocess.run(["blender", "-b", "-t", "8", "--python", str(BLENDER_SCRIPT), "--", str(npz), str(a.out / f"k{i:02d}")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tiles.append(a.out / f"k{i:02d}" / "00000.png")
    (a.out / "labels.json").write_text(json.dumps(labels, indent=1))
    print("\n".join(labels))
    # sheet: 4 columns, tiles 480x360, label on top
    inputs, filt = [], []
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        src = t if t is not None else None
        if src is None:
            inputs += ["-f", "lavfi", "-i", "color=c=gray:s=640x480:d=1"]
        else:
            inputs += ["-i", str(src)]
        status = lab.split("|")[-1].strip() if "|" in lab else "FAIL"
        text = f"{k} {lab.split(':')[1].split('|')[0].strip()} {status}"
        text = "".join(ch for ch in text if ch.isalnum() or ch in " ._-")  # drawtext-safe
        filt.append(f"[{k}]scale=480:360,drawtext=fontfile={FONT}:text='{text}':x=8:y=8:fontsize=18:fontcolor=black[v{k}]")
    n = len(tiles)
    cols = 4
    rows = (n + cols - 1) // cols
    while len(filt) < rows * cols:
        k = len(filt)
        inputs += ["-f", "lavfi", "-i", "color=c=white:s=480x360:d=1"]
        filt.append(f"[{k}]null[v{k}]")
    layout = "|".join(f"{c * 480}_{r * 360}" for r in range(rows) for c in range(cols))
    graph = ";".join(filt) + ";" + "".join(f"[v{k}]" for k in range(rows * cols)) + f"xstack=inputs={rows * cols}:layout={layout}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-frames:v", "1", str(a.out / "sheet.png")], check=True)
    print(f"sheet: {a.out / 'sheet.png'}")


if __name__ == "__main__":
    main()
