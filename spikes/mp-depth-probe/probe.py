"""Run MediaPipe's pose landmark model (from the Holistic bundle) on rendered contrast frames and keep
its outputs and intermediate activations.

    uv run python probe.py run            # data/spikes/mp-depth/configs/c*/frames -> features.npz
    uv run python probe.py check <png>    # input range / crop sanity on one frame

Crop: MediaPipe's tracking ROI from its own auxiliary landmarks 33 (centre) and 34 (scale/rotation
point): square of side 2 * |p34 - p33| * 1.25, rotated so p33 -> p34 points up. Starting from a guess
around the true hips and shoulders, the crop is refined from the model's own output until it settles
(as frame-to-frame tracking would). Out-of-image pixels are zero.

Per frame we keep: image landmarks (39, 5) in image pixels (x, y, z scaled like x), world landmarks
(39, 3) m, the shared head features `re_lu_55` (768: what both landmark heads read, linearly), and for
every activation layer the feature vector at the true hand and nose positions (nearest cell) and the
spatial mean.
"""

import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "spikes" / "mp-depth"
MODEL = D / "model" / "x" / "pose_landmarks_detector.tflite"
NOSE, LWR, RWR, LPI, RPI, LIX, RIX = 0, 15, 16, 17, 18, 19, 20
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
HEAD = "re_lu_55"


class Model:
    def __init__(self):
        self.fast = Interpreter(model_path=str(MODEL))
        self.fast.allocate_tensors()
        self.slow = Interpreter(model_path=str(MODEL), experimental_preserve_all_tensors=True,
                                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.slow.allocate_tensors()
        self.inp = self.fast.get_input_details()[0]["index"]
        self.out = {d["name"]: d["index"] for d in self.fast.get_output_details()}
        # activation layers (4D, after ReLU6), in graph order
        T = {t["index"]: t for t in self.slow.get_tensor_details()}
        self.layers = []
        for o in self.slow._get_ops_details():
            for i in o["outputs"]:
                t = T[i]
                if len(t["shape"]) == 4 and "re_lu" in t["name"] and t["shape"][1] > 1:
                    short = t["name"].split("/")[2].split(";")[0]
                    if short not in [s for s, _, _ in self.layers]:
                        self.layers.append((short, i, tuple(t["shape"][1:])))

    def run(self, x, deep=False):
        it = self.slow if deep else self.fast
        it.set_tensor(self.inp, x[None].astype(np.float32))
        it.invoke()
        o = {k: it.get_tensor(v)[0] for k, v in self.out.items()}
        acts = {s: it.get_tensor(i)[0] for s, i, _ in self.layers} if deep else None
        return o, acts


def crop(img, c, size, theta, n=256):
    """Square crop centred c (px), side size (px), rotated theta (crop up = image direction
    (sin theta, -cos theta)); bilinear, zero outside. Returns (n, n, 3) and the crop->image map."""
    H, W, _ = img.shape
    g = (np.arange(n) + 0.5) / n - 0.5
    u, v = np.meshgrid(g, g)
    cs, sn = np.cos(theta), np.sin(theta)
    x = c[0] + size * (cs * u - sn * v)
    y = c[1] + size * (sn * u + cs * v)
    x0, y0 = np.floor(x - 0.5).astype(int), np.floor(y - 0.5).astype(int)
    fx, fy = x - 0.5 - x0, y - 0.5 - y0
    out = np.zeros((n, n, 3))
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out += np.where(ok[..., None], img[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)], 0) * w[..., None]

    def to_img(p):  # crop pixels (..., 2) -> image pixels
        uu, vv = p[..., 0] / n - 0.5, p[..., 1] / n - 0.5
        return np.stack([c[0] + size * (cs * uu - sn * vv), c[1] + size * (sn * uu + cs * vv)], -1)

    def to_crop(p):  # image pixels -> crop pixels
        dx, dy = (p[..., 0] - c[0]) / size, (p[..., 1] - c[1]) / size
        return np.stack([(cs * dx + sn * dy + 0.5) * n, (-sn * dx + cs * dy + 0.5) * n], -1)
    return out, to_img, to_crop


def roi_from(p33, p34):
    v = p34 - p33
    return p33, 2 * np.linalg.norm(v) * 1.25, np.arctan2(v[0], -v[1])


def track(model, img, guess, iters=4, scale=1.0):
    """Refine the ROI from the model's own aux landmarks; returns final outputs and crop maps."""
    c, size, th = guess
    for _ in range(iters):
        x, to_img, to_crop = crop(img, c, size, th)
        o, _ = model.run(x * scale)
        lm = o["Identity"].reshape(39, 5)
        p = to_img(lm[:, :2])
        c, size, th = roi_from(p[33], p[34])
    return (c, size, th)


def initial_guess(p2):
    hip = 0.5 * (p2[L_HIP] + p2[R_HIP])
    sh = 0.5 * (p2[L_SH] + p2[R_SH])
    return hip, 2 * np.linalg.norm(sh - hip) * 1.25 * 1.4, np.arctan2(*(sh - hip) * [1, -1])


def run_all(scale=1.0):
    model = Model()
    rows = []
    for d in sorted((D / "configs").glob("c*")):
        meta = np.load(d / "meta.npz")
        for g in range(len(meta["gaps"])):
            f = d / "frames" / f"{g:05d}.png"
            if not f.exists():
                continue
            img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
            H, W, _ = img.shape
            p2 = meta["pose2d"][g, :, :2] * [W, H]
            roi = track(model, img, initial_guess(p2), scale=scale)
            x, to_img, to_crop = crop(img, *roi)
            o, acts = model.run(x * scale, deep=True)
            lm = o["Identity"].reshape(39, 5)
            img_lm = np.concatenate([to_img(lm[:, :2]), lm[:, 2:3] * roi[1] / 256, lm[:, 3:]], -1)
            side = str(meta["side"])
            hand_px = 0.5 * (p2[LPI if side == "l" else RPI] + p2[LIX if side == "l" else RIX])
            pts = to_crop(np.stack([hand_px, p2[NOSE]])) / 256  # crop fraction
            loc = {}
            for s, _, shp in model.layers:
                a = acts[s]
                ij = np.clip((pts[:, ::-1] * shp[:2]).astype(int), 0, np.array(shp[:2]) - 1)
                loc[s] = np.stack([a[ij[0, 0], ij[0, 1]], a[ij[1, 0], ij[1, 1]], a.mean((0, 1))])  # hand, nose, mean
            rows.append(dict(config=d.name, gap=meta["gaps"][g], side=side, face_point=str(meta["face_point"]),
                             palm=bool(meta["palm"]), cam_xyz=meta["cam_xyz"][g], img_lm=img_lm, world=o["Identity_4"].reshape(39, 3),
                             flag=float(o["Identity_1"][0]), head=acts[HEAD].reshape(-1), loc=loc, roi=np.array([*roi[0], roi[1], roi[2]]),
                             crop_hand=pts[0]))
        print(d.name, "ok", flush=True)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "loc"}
    for s, _, _ in model.layers:
        out[f"loc/{s}"] = np.stack([r["loc"][s] for r in rows])
    out["layers"] = np.array([s for s, _, _ in model.layers])
    out["layer_shapes"] = np.array([shp for _, _, shp in model.layers])
    np.savez_compressed(D / "features.npz", **out)
    print(f"{len(rows)} frames, {len(model.layers)} layers -> {D / 'features.npz'}")


def check(png):
    model = Model()
    img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255.0
    H, W, _ = img.shape
    d = Path(png).parents[1]
    meta = np.load(d / "meta.npz")
    g = int(Path(png).stem)
    p2 = meta["pose2d"][g, :, :2] * [W, H]
    for scale, label in ((1.0, "[0,1]"),):
        roi = track(model, img, initial_guess(p2), scale=scale)
        x, to_img, _ = crop(img, *roi)
        o, _ = model.run(x * scale)
        lm = to_img(o["Identity"].reshape(39, 5)[:, :2])
        err = np.linalg.norm(lm[:33] - p2, axis=-1)
        print(f"{label}: pose flag {o['Identity_1'][0]:.3f}, roi {roi[0].round(1)} size {roi[1]:.0f} rot {np.degrees(roi[2]):.1f}, "
              f"2D error vs true (px) median {np.median(err):.1f}, nose {err[0]:.1f}, wrists {err[15]:.1f}/{err[16]:.1f}")
        Image.fromarray((x * 255).astype(np.uint8)).save(D / "check_crop.png")
    print("layers:", len(model.layers), [(s, shp) for s, _, shp in model.layers[::8]])


def draw(png, out):
    from PIL import ImageDraw
    model = Model()
    img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64) / 255.0
    H, W, _ = img.shape
    meta = np.load(Path(png).parents[1] / "meta.npz")
    p2 = meta["pose2d"][int(Path(png).stem), :, :2] * [W, H]
    roi = track(model, img, initial_guess(p2))
    x, to_img, _ = crop(img, *roi)
    lm = to_img(model.run(x)[0]["Identity"].reshape(39, 5)[:, :2])
    im = Image.open(png).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k, (a, b) in enumerate(zip(p2, lm[:33])):
        col = (255, 0, 0) if k in (15, 17, 19) else (0, 0, 255) if k in (16, 18, 20) else (0, 160, 0)
        dr.ellipse([*(b - 4), *(b + 4)], outline=col, width=2)  # predicted: circle
        dr.rectangle([*(a - 2), *(a + 2)], fill=col)  # true: square
    im.save(out)


if __name__ == "__main__":
    if sys.argv[1] == "draw":
        draw(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run_all()
    else:
        check(sys.argv[2])

