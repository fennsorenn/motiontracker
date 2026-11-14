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
  (0, with a dead zone of twice the open-eye noise) to fully closed at 3.5 times the noise (a clear blink; see the
  BLINK_* comment);
- squint: per eye, a resting baseline that follows drops quickly and rises slowly; squint is rescaled above it;
- linking: blink comes from both eyes' mean excursion, applied to both, unless one eye is closed while the other
  stays at its open level (a wink); squint: when the two eyes' values are close, both get their mean.
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
# blink as an event: at ~10 Hz most blinks are caught in one frame part-way closed (owner's sessions: raw peaks
# mostly 0.5-0.6 seated, 0.4-0.6 in the hands session, i.e. only 0.2-0.3 above the open level there; a fixed
# closed level of 0.7 left most blinks at 0.3-0.6 of a closure). So the excursion e above the open level is
# measured against the open-eye noise (sd of the residual around the fitted open level, tracked online, ~0.06):
# below BLINK_DEADZONE_SD x noise the eye reads open (no flutter), above BLINK_EVENT_SD x noise it reads fully
# closed, smoothstep in between (half-closed eyes still show partly).
BLINK_NOISE_PRIOR = 0.06
BLINK_EVENT_SD, BLINK_DEADZONE_SD = 3.5, 2.0
# the eyes blink together: MediaPipe often catches a blink strongly in one eye and only partly in the other (hands
# session: 41 frames with one eye closed and the other below its dead zone, though raised 0.1-0.2; 0 frames with
# both closed). So both eyes get the blink of their mean excursion (noise sd of a mean of two: /sqrt 2), unless it
# is a wink: one eye closed while the other is below BLINK_WINK_SD x its noise above its open level.
BLINK_WINK_SD = 0.5


def _blink_curve(e, sd):
    """Excursion above the open level -> 0 (open) .. 1 (closed): dead zone, then smoothstep to the event level."""
    lo, hi = BLINK_DEADZONE_SD * sd, max(BLINK_EVENT_SD * sd, BLINK_DEADZONE_SD * sd + 0.05)
    u = _clamp((e - lo) / (hi - lo))
    return u * u * (3 - 2 * u)
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
        self.noise_var = {s: BLINK_NOISE_PRIOR ** 2 for s in SIDES}
        self.squint_base = {s: None for s in SIDES}
        self.t = None

    def step(self, t, shapes):
        """t (s), shapes {ARKit name: raw 0..1} -> processed copy."""
        if not shapes:
            return shapes
        dt = 0.1 if self.t is None else _clamp(t - self.t, 0.0, 1.0)
        self.t = t
        out = dict(shapes)
        blink, squint, exc = {}, {}, {}
        for s in SIDES:
            b, ld, q = shapes.get(f"eyeBlink{s}"), shapes.get(f"eyeLookDown{s}", 0.0), shapes.get(f"eyeSquint{s}")
            if b is not None:
                fit = self.fit[s]
                o = _clamp(fit.level(ld), 0.0, 0.6)
                e = b - o
                sd = self.noise_var[s] ** 0.5
                exc[s] = (e, sd)
                blink[s] = _blink_curve(e, sd)
                if b < o + BLINK_GATE:
                    fit.update(ld, b, dt)
                    f = 1 - pow(2.718281828, -dt / BLINK_TAU_S)
                    self.noise_var[s] += (e * e - self.noise_var[s]) * f
            if q is not None:
                base = self.squint_base[s]
                if base is None:
                    base = q
                tau = SQUINT_DOWN_S if q < base else SQUINT_UP_S
                base += (q - base) * (1 - pow(2.718281828, -dt / tau))
                self.squint_base[s] = base
                squint[s] = _clamp((q - base) / max(1.0 - base, 1e-3))
        if len(exc) == 2:  # blink from both eyes together (see BLINK_WINK_SD)
            (el, sl), (er, sr) = exc["Left"], exc["Right"]
            both = _blink_curve(0.5 * (el + er), 0.5 * (sl * sl + sr * sr) ** 0.5)
            wl = blink["Left"] >= 0.9 and er < BLINK_WINK_SD * sr
            wr = blink["Right"] >= 0.9 and el < BLINK_WINK_SD * sl
            if not (wl or wr):
                blink = {"Left": both, "Right": both}
        for name, vals in (("eyeBlink", blink), ("eyeSquint", squint)):
            if len(vals) == 2 and name == "eyeSquint":
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
