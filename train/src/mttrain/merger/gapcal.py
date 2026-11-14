"""Body model sim-to-real gap per joint (merger calibration, `batch.degrade`): m1-v10's global errors on real Holistic
output from the rendered sets against its errors on synthetic merger batches -> data/merger/calib/m1_gap.json.

    uv run python -m mttrain.merger.gapcal <merger pool dir with val.npz> [<body model checkpoint, default m1-v10>]

m1-v12+ (legs, root and up): also the thighs and shins, the root position (image plane and depth, cm) and the up
direction (deg). Writes `batch.gap_file(<checkpoint>)`.
"""
import json, sys, numpy as np, torch
from pathlib import Path
from mtcore.mhr import load, skeleton_state
from mttrain.gpugen.geom import quat_to_mat
from mttrain.infer import Runner
from mttrain.merger.batch import MergerBatches, gap_file, geodesic, globals_from_locals
from mttrain.skeleton import OUT_JOINTS, TargetExtractor
M1 = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../data/m1/runs/m1-v10/best.pt")
J = ["c_head", "l_uparm", "l_lowarm", "l_wrist", "r_uparm", "r_lowarm", "r_wrist", "c_spine3"]
LEGS = ["l_upleg", "l_lowleg", "r_upleg", "r_lowleg"]
ch = load(); pn = list(ch.parameter_transform.names); jn = list(ch.skeleton.joint_names)
runner = Runner(M1)
if runner.model.n_out > 17:
    J += LEGS
te = TargetExtractor(jn, skeleton_state(ch, np.zeros(len(pn))))
RX = ["root_xy_cm", "root_z_cm", "root_logz", "up_deg"]  # root_logz: |log(predicted / true depth)|
rest = skeleton_state(ch, np.zeros(len(pn)))
idxJ = [jn.index(j) for j in OUT_JOINTS]
rest_g = quat_to_mat(torch.tensor(rest[idxJ, 3:7], dtype=torch.float32))
real = {j: [] for j in J + RX}
for setd in ["../data/m1/renders/v1", "../data/m1/renders/rest1", "../data/m1/renders/contact1", "../data/m1/renders/head1",
             "../data/m4/renders/pairs3", "../data/m4/renders/hands3", "../data/m4/renders/pairs2", "../data/m4/renders/hands2"]:
    for d in sorted(Path(setd).glob("seq*")):
        z = np.load(d / "seq.npz"); fps = float(z["fps"])
        rows = [json.loads(l) for l in open(d / "landmarks.jsonl") if l.strip()]
        T = len(rows); times = np.array([r["t"] for r in rows]) / 1000
        world, image, vis = np.zeros((T, 33, 3)), np.zeros((T, 33, 3)), np.zeros((T, 33)); present = np.zeros(T, bool)
        for i, f in enumerate(rows):
            if f.get("pose") and f.get("pose2d"):
                p = np.asarray(f["pose"]); world[i], vis[i], image[i] = p[:, :3], p[:, 3], np.asarray(f["pose2d"])[:, :3]; present[i] = True
        rot, _ = runner.predict(times, world, image, vis, present)
        k = np.clip(np.round(times * fps).astype(int), 0, len(z["mp"]) - 1)
        st = skeleton_state(ch, z["mp"][k].astype(np.float64))[:, idxJ]
        Gt = torch.tensor(z["R"], dtype=torch.float32) @ quat_to_mat(torch.tensor(st[..., 3:7], dtype=torch.float32)) @ rest_g.transpose(-1, -2)
        G = globals_from_locals(torch.tensor(rot, dtype=torch.float32))
        for j in J:
            e = torch.rad2deg(geodesic(G[:, OUT_JOINTS.index(j)], Gt[:, OUT_JOINTS.index(j)])).numpy()
            real[j].append(e[present])
        if runner.aux is not None:
            rt, ut = te.root(skeleton_state(ch, z["mp"][k].astype(np.float64)), z["R"], z["t"])
            d = (runner.aux["root"] - rt) * 100
            real["root_xy_cm"].append(np.linalg.norm(d[:, :2], axis=-1)[present])
            real["root_z_cm"].append(np.abs(d[:, 2])[present])
            real["root_logz"].append(np.abs(np.log(np.clip(runner.aux["root"][:, 2], 0.05, None) / rt[:, 2]))[present])
            real["up_deg"].append(np.degrees(np.arccos(np.clip((runner.aux["up"] * ut).sum(-1), -1, 1)))[present])
mb = MergerBatches("val", "cuda", seed=9, pool=Path(sys.argv[1]), m1=M1)
syn = {j: [] for j in J + RX}
for _ in range(8):
    b = mb.sample(128, gap=False)
    if "root" in b["body"]:
        d = (b["body"]["root"] - b["target"]["root"]) * 100
        syn["root_xy_cm"].append(d[..., :2].norm(dim=-1).flatten().cpu().numpy())
        syn["root_z_cm"].append(d[..., 2].abs().flatten().cpu().numpy())
        syn["root_logz"].append(torch.log(b["body"]["root"][..., 2].clamp_min(0.05) / b["target"]["root"][..., 2]).abs().flatten().cpu().numpy())
        syn["up_deg"].append(torch.rad2deg(torch.acos((b["body"]["up"] * b["target"]["up"]).sum(-1).clamp(-1, 1))).flatten().cpu().numpy())
    G, Gt = globals_from_locals(b["body"]["rot"]), globals_from_locals(b["target"]["rot"])
    for j in J:
        syn[j].append(torch.rad2deg(geodesic(G[..., OUT_JOINTS.index(j), :, :], Gt[..., OUT_JOINTS.index(j), :, :])).flatten().cpu().numpy())
out = {}
for j in J + [k for k in RX if real[k]]:
    r, s = np.concatenate(real[j]), np.concatenate(syn[j])
    out[j] = {"real_p50": float(np.median(r)), "syn_p50": float(np.median(s)), "real_p90": float(np.percentile(r, 90)), "syn_p90": float(np.percentile(s, 90))}
    print(f"{j:9s} real p50 {out[j]['real_p50']:5.1f} p90 {out[j]['real_p90']:5.1f}   synthetic p50 {out[j]['syn_p50']:5.1f} p90 {out[j]['syn_p90']:5.1f}")
json.dump(out, open(gap_file(M1), "w"), indent=1)
print("->", gap_file(M1))
