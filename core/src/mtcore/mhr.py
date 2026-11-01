"""MHR rig loading and forward kinematics via pymomentum.

Uses pymomentum directly: MHR.from_files segfaults in Character.with_blend_shape
(pymomentum-cpu 0.1.114, torch 2.8 CPU), and the skeleton never depends on the
identity blendshapes anyway (see mhr/mhr.py forward: skel_state comes from model
parameters only).
"""

from pathlib import Path

import numpy as np
import pymomentum.geometry as g

from .paths import MHR_ASSETS

MODEL_FILE = MHR_ASSETS / "compact_v6_1.model"
# pose parameters that translate joints; VRM bones only rotate, so these stay zero
FLEX_T = {"spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible", "arm_length_flexible",
          "hip_width_flexible", "leg_length_flexible"}


def load(lod=1):
    return g.Character.load_fbx(str(MHR_ASSETS / f"lod{lod}.fbx"), str(MODEL_FILE), load_blendshapes=False)


def skeleton_state(ch, mp):
    """Model parameters (..., P) -> skeleton state (..., J, 8): translation, quaternion xyzw, scale."""
    return np.asarray(g.model_parameters_to_skeleton_state(ch, np.asarray(mp, dtype=np.float32)))


def positions(ch, mp):
    return skeleton_state(ch, mp)[..., :3]


def limits(path=MODEL_FILE):
    """Parameter name -> (min, max) from the model file's [Limits] minmax entries (soft penalties in MHR)."""
    lim = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] == "limit" and p[2] == "minmax":
            lo, hi = line.split("[")[1].split("]")[0].split(",")
            lim[p[1]] = (float(lo), float(hi))
    return lim
