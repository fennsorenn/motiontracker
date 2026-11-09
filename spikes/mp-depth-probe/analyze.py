"""Where does MediaPipe's pose landmark model represent hand-vs-face depth?

    uv run python analyze.py

Target: true depth of the active hand (middle of index and pinky knuckles) minus the nose, camera
frame, cm (negative = hand nearer the camera). Each configuration has the hand at 4 gaps from the face
(0.5, 4, 8, 14 cm along the face normal), everything else fixed.

1. Outputs: how MediaPipe's own hand-minus-nose depth (world, and image z) follows the true one.
   "within" = after removing each configuration's mean: does it track the gap at all?
2. Per layer: ridge regression from the activations at the hand and the nose (+ spatial mean) to the
   true depth, 5-fold cross-validated with whole configurations held out (alpha by inner CV).
   R2 overall and within configurations.
3. The shared head features (768; both landmark heads are linear in them): same probe. If depth is
   linearly readable here, a closed-form edit of the world head's hand rows can fix it.
4. Contrast consistency per layer: per configuration, activation change at the hand from contact to
   14 cm; mean pairwise cosine between configurations (1 = one shared depth direction).
"""

from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[2] / "data" / "spikes" / "mp-depth"
NOSE, LPI, RPI, LIX, RIX = 0, 17, 18, 19, 20


def hand_minus_nose(P, side):
    h = np.where((side == "l")[:, None], 0.5 * (P[:, LPI] + P[:, LIX]), 0.5 * (P[:, RPI] + P[:, RIX]))
    return h - P[:, NOSE]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-3 * max(np.median(sd[sd > 0]) if (sd > 0).any() else 1.0, 1e-9)  # drop dead channels
    X, mu, sd = X[:, keep], mu[keep], sd[keep]
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym)) if Z.shape[1] <= len(Z) else \
        Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(len(Z)), y - ym)
    return lambda Xn: ((Xn[:, keep] - mu) / sd) @ w + ym


def folds(groups, k, seed=0):
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    return [np.isin(groups, part) for part in np.array_split(u, k)]


ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


def cv_predict(X, y, groups, k=5):
    pred = np.zeros_like(y)
    for test in folds(groups, k):
        tr = ~test
        best, ba = np.inf, ALPHAS[0]
        for a in ALPHAS:  # inner CV on the training configurations
            err = 0.0
            for itest in folds(groups[tr], 4, seed=1):
                f = ridge_fit(X[tr][~itest], y[tr][~itest], a)
                err += ((f(X[tr][itest]) - y[tr][itest]) ** 2).sum()
            if err < best:
                best, ba = err, a
        pred[test] = ridge_fit(X[tr], y[tr], ba)(X[test])
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def within(v, groups):
    out = v.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = v[m] - v[m].mean()
    return out


def main():
    F = np.load(D / "features.npz", allow_pickle=True)
    side, groups, gap = F["side"], F["config"], F["gap"]
    y = hand_minus_nose(F["cam_xyz"], side)[:, 2] * 100  # cm
    n = len(y)
    print(f"{n} frames, {len(np.unique(groups))} configurations; true hand-nose depth cm: "
          f"at contact median {np.median(y[gap == 0.5]):+.1f}, at 14 cm {np.median(y[gap == 14]):+.1f}")

    print("\n== 1. MediaPipe's own outputs (hand minus nose depth)")
    mw = hand_minus_nose(F["world"], side)[:, 2] * 100
    # image z is in pixels scaled like x: convert with the true shoulder width (m) / its image length (px)
    imz = hand_minus_nose(F["img_lm"][..., :3], side)[:, 2]
    print(f"{'gap cm':>8s} {'true':>7s} {'MP world':>9s} {'MP image z (px)':>16s}")
    for g in np.unique(gap):
        m = gap == g
        print(f"{g:8.1f} {np.median(y[m]):+7.1f} {np.median(mw[m]):+9.1f} {np.median(imz[m]):+16.1f}")
    for name, v in (("world", mw), ("image z", imz)):
        yw, vw = within(y, groups), within(v, groups)
        slope = (yw * vw).sum() / (yw * yw).sum()
        print(f"  {name}: within-configuration slope vs truth {slope:+.2f}, correlation {np.corrcoef(yw, vw)[0, 1]:+.2f}")

    print("\n== 2./3. Linear probes: R2 overall / within configurations (5-fold, configurations held out)")
    layers = list(F["layers"])
    shapes = F["layer_shapes"]
    rows = []
    for s, shp in zip(layers, shapes):
        L = F[f"loc/{s}"]  # (n, 3, C): hand, nose, mean
        X = L.reshape(n, -1)
        p = cv_predict(X, y, groups)
        rows.append((s, tuple(shp), r2(y, p), r2(within(y, groups), within(p, groups))))
    p = cv_predict(F["head"], y, groups)
    rows.append(("HEAD re_lu_55 (768)", (2, 2, 192), r2(y, p), r2(within(y, groups), within(p, groups))))
    head_pred = p
    # controls: shuffled target on the head features; MediaPipe's own world output as a 1-D "probe"
    rng = np.random.default_rng(0)
    ys = rng.permutation(y)
    p = cv_predict(F["head"], ys, groups)
    rows.append(("control: head, shuffled target", (2, 2, 192), r2(ys, p), float("nan")))
    p = cv_predict(mw[:, None], y, groups)
    rows.append(("control: MP world dz, linear", (1,), r2(y, p), r2(within(y, groups), within(p, groups))))
    for s, shp, a, b in rows:
        print(f"  {s:32s} {str(shp):16s} {a:+.2f} / {b:+.2f}")

    print("\n== 3b. Head probe error (cm), cross-validated, by gap")
    for g in np.unique(gap):
        m = gap == g
        print(f"  gap {g:4.1f}: median |error| probe {np.median(np.abs(head_pred[m] - y[m])):4.1f}, "
              f"MediaPipe world {np.median(np.abs(mw[m] - y[m])):4.1f}")

    print("\n== 4. Contrast consistency at the hand (mean pairwise cosine of contact -> 14 cm changes)")
    for s in layers[::4] + [layers[-1]]:
        L = F[f"loc/{s}"][:, 0]
        diffs = []
        for g in np.unique(groups):
            m = groups == g
            a, b = L[m & (gap == 0.5)], L[m & (gap == 14)]
            if len(a) and len(b):
                diffs.append(b[0] - a[0])
        Dm = np.array(diffs)
        Dm = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-9)
        C = Dm @ Dm.T
        print(f"  {s:12s} {np.mean(C[np.triu_indices(len(C), 1)]):+.2f}")


if __name__ == "__main__":
    main()
