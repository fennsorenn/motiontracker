import numpy as np
import pytest

from mtcore.paths import AVATARS, CMU_ROOT, MHR_ASSETS

needs_mhr = pytest.mark.skipif(not (MHR_ASSETS / "lod1.fbx").exists(), reason="MHR assets not in data/mhr/assets")
needs_cmu = pytest.mark.skipif(not CMU_ROOT.exists(), reason="CMU data not in data/cmu")
needs_avatars = pytest.mark.skipif(not AVATARS.exists(), reason="VRoid samples (../volleyball) not found")


@pytest.fixture(scope="session")
def mhr():
    from mtcore.mhr import load
    ch = load()
    return ch, list(ch.skeleton.joint_names), list(ch.parameter_transform.names)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
