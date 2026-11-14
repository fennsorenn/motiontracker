"""Face blendshape post-processing for the live runtime and the avatar videos (plain Python, causal, per frame).

MediaPipe's blendshapes come at the detector rate (~10 Hz) and, measured on the owner's sessions (2026-11-14):
- eyeBlink rests at 0.25-0.41 with the eyes open and rises with eyeLookDown (correlation ~0.7; blink ~ 0.15-0.22
  + 0.3-0.46 lookDown): looking down at the screen reads as a half-closed eye;
- eyeSquint rests at 0.34-0.45 (and vspark's ARKit -> VRoid table adds 0.3 x squint to eye closure);
- the two eyes differ (left blink 0.41 vs right 0.32 in the hands session, correlation 0.37).
So the avatar's eyes sat ~40% closed and lopsided, and the face stepped at 10 Hz.

`EyeFilter` (per frame, causal):
- blink: per eye, the open-eye level is tracked as a linear function of that eye's lookDown, fitted online with
  exponential forgetting (BLINK_TAU_S) on frames that don't look like a blink; blink is rescaled from that level
  (0) to BLINK_CLOSED (1);
- squint: per eye, a resting baseline that follows drops quickly and rises slowly; squint is rescaled above it;
- linking: when the two eyes' processed values are close, both get their mean (a wink, a clear difference, passes).
Other shapes pass through unchanged.

`Glide`: fills the frames between detector frames without smoothing, like the harness's bone glide
(`mteval.upsample`): on a new frame every shape moves linearly from what is shown to the new value, finishing one
output tick before the next frame is expected (expected interval: mean of the last GLIDE_WINDOW intervals).
"""

from collections import deque

SIDES = ("Left", "Right")

BLINK_TAU_S = 60.0  # forgetting time constant of the open-eye fit
BLINK_PRIOR = (0.18, 0.4)  # open-eye level = a + k * lookDown before data (owner's sessions: 0.15-0.22, 0.31-0.46)
BLINK_PRIOR_S = 5.0  # weight of the prior, in seconds of data
BLINK_GATE = 0.12  # frames more than this above the fitted open level are not used for the fit (blinks)
BLINK_CLOSED = 0.7  # raw blink of a closed eye (MediaPipe peaks ~0.6-0.75)
BLINK_SPAN_MIN = 0.25  # closed level at least this far above the open level
SQUINT_DOWN_S, SQUINT_UP_S = 1.0, 20.0  # baseline time constants: follow drops fast, rises slowly
LINK_START, LINK_FULL = 0.15, 0.35  # |left - right| below LINK_START: fully linked; above LINK_FULL: independent
GLIDE_WINDOW = 10


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


class _OpenFit:
    """Weighted least squares of blink on lookDown with exponential forgetting, starting from a prior."""

    def __init__(self):
        a, k = BLINK_PRIOR
        w = BLINK_PRIOR_S
        # prior as two pseudo-points at lookDown 0 and 0.6, w/2 seconds each
        self.s = [0.0] * 5  # sum w, w x, w y, w x^2, w x y
        for x in (0.0, 0.6):
            self._add(x, a + k * x, w / 2)

    def _add(self, x, y, w):
        s = self.s
        s[0] += w; s[1] += w * x; s[2] += w * y; s[3] += w * x * x; s[4] += w * x * y

    def level(self, x):
        n, sx, sy, sxx, sxy = self.s
        det = n * sxx - sx * sx
        if det <= 1e-9:
            return sy / max(n, 1e-9)
        k = (n * sxy - sx * sy) / det
        a = (sy - k * sx) / n
        return a + k * x

    def update(self, x, y, dt):
        f = pow(2.718281828, -dt / BLINK_TAU_S)
        self.s = [v * f for v in self.s]
        self._add(x, y, dt)


class EyeFilter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fit = {s: _OpenFit() for s in SIDES}
        self.squint_base = {s: None for s in SIDES}
        self.t = None

    def step(self, t, shapes):
        """t (s), shapes {ARKit name: raw 0..1} -> processed copy."""
        if not shapes:
            return shapes
        dt = 0.1 if self.t is None else _clamp(t - self.t, 0.0, 1.0)
        self.t = t
        out = dict(shapes)
        blink, squint = {}, {}
        for s in SIDES:
            b, ld, q = shapes.get(f"eyeBlink{s}"), shapes.get(f"eyeLookDown{s}", 0.0), shapes.get(f"eyeSquint{s}")
            if b is not None:
                fit = self.fit[s]
                o = _clamp(fit.level(ld), 0.0, 0.6)
                c = max(BLINK_CLOSED, o + BLINK_SPAN_MIN)
                blink[s] = _clamp((b - o) / (c - o))
                if b < o + BLINK_GATE:
                    fit.update(ld, b, dt)
            if q is not None:
                base = self.squint_base[s]
                if base is None:
                    base = q
                tau = SQUINT_DOWN_S if q < base else SQUINT_UP_S
                base += (q - base) * (1 - pow(2.718281828, -dt / tau))
                self.squint_base[s] = base
                squint[s] = _clamp((q - base) / max(1.0 - base, 1e-3))
        for name, vals in (("eyeBlink", blink), ("eyeSquint", squint)):
            if len(vals) == 2:
                l, r = vals["Left"], vals["Right"]
                w = _clamp((abs(l - r) - LINK_START) / (LINK_FULL - LINK_START))
                m = 0.5 * (l + r)
                vals = {"Left": w * l + (1 - w) * m, "Right": w * r + (1 - w) * m}
            for s, v in vals.items():
                out[f"{name}{s}"] = v
        return out


class Glide:
    """Smoothing-free upsampling of per-frame shape values to an output rate (see the module docstring)."""

    def __init__(self, hz=60.0):
        self.tick = 1.0 / hz
        self.reset()

    def reset(self):
        self.shown, self.start, self.target = {}, {}, {}
        self.t_new, self.dur = None, self.tick
        self.times = deque(maxlen=GLIDE_WINDOW + 1)

    def push(self, t, shapes):
        """A new frame at time t (s); shapes None or {} keep the last values."""
        if not shapes:
            return
        self.times.append(t)
        if len(self.times) > 1:
            ts = list(self.times)
            mean = (ts[-1] - ts[0]) / (len(ts) - 1)
            self.dur = max(self.tick, mean - self.tick)
        self.start = dict(self.shown) if self.shown else dict(shapes)
        self.target = dict(shapes)
        self.t_new = t

    def at(self, t):
        """Values shown at time t (s), {} before the first frame."""
        if self.t_new is None:
            return {}
        u = _clamp((t - self.t_new) / self.dur)
        self.shown = {n: self.start.get(n, v) + (v - self.start.get(n, v)) * u for n, v in self.target.items()}
        return self.shown
