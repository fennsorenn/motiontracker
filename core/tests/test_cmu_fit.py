import numpy as np

from conftest import needs_cmu, needs_mhr
from mtcore.cmu_fit import fit


@needs_mhr
@needs_cmu
def test_fit_short_clip():
    _, mp, _, res = fit(2, 1, step=8, max_frames=40, verbose=False)
    assert mp.shape[0] == 40
    assert np.median(res) < 2.0  # cm; spike: 0.4-1.3 median per clip
    assert np.isfinite(mp).all()
