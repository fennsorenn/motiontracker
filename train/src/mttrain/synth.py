"""Fitted MHR motion -> synthetic MediaPipe Pose observations (plan steps 3-4)."""

import numpy as np

from mtcore.mhr import skeleton_state

from .camera import sample_desk_camera
from .mpstyle import Style
from .noise import NoiseConfig, observe


def frame_times(rng, duration, fps=10.0, drop_rate=0.03):
    """Detector frame times (s) over [0, duration): nominal fps with small jitter and dropped frames.

    Session: 100 ms median interval, occasional gaps up to ~270 ms."""
    step = 1.0 / fps
    t = np.arange(0.0, duration, step) + rng.normal(0, 0.004, int(np.ceil(duration / step)))[: int(np.ceil(duration / step))]
    t = np.clip(np.sort(t), 0, duration - 1e-6)
    keep = rng.random(len(t)) >= drop_rate
    keep[0] = True
    return t[keep]


def synthesize(rng, ch, landmark_model, mp, src_fps=30.0, fps=None, style=None, noise=None, camera=None):
    """One observed sequence from model parameters mp (N, P) sampled at src_fps.

    Returns dict: t (T,) s, states (T, J, 8) MHR skeleton states (targets), world/image (T, 33, 3)
    noisy MediaPipe-style landmarks, vis (T, 33), present (T,), clean_world, clean_image."""
    fps = fps or float(rng.uniform(8, 30))
    t = frame_times(rng, len(mp) / src_fps, fps)
    idx = np.minimum(np.round(t * src_fps).astype(int), len(mp) - 1)
    states = skeleton_state(ch, mp[idx])
    X = landmark_model(states) / 100.0  # MHR world, metres
    style = style or Style.sample(rng)
    X = style.placement(X)
    cam = camera or sample_desk_camera(rng, X)
    image, _ = cam.project(X)
    world = style.world(cam.world_landmarks(X))
    dt = np.diff(t, prepend=t[0] - 1.0 / fps)
    W, Im, vis, present = observe(rng, world, image, dt, noise or NoiseConfig.sample(rng))
    image = image.copy()
    image[..., 2] *= style.image_z  # MediaPipe's exaggerated image depth; scales its noise too
    Im[..., 2] *= style.image_z
    return dict(t=t, states=states, world=W, image=Im, vis=vis, present=present,
                clean_world=world, clean_image=image, camera=cam)
