"""History picking by age in the live merger (mtcore.merged): steps, shifts, pruning."""
import pytest

from mtcore.merged import pick_history, prune_history, shifted_ages


def test_shifted_ages():
    ages = [0.0, 0.1, 0.2, 0.4, 0.8]
    assert shifted_ages(ages, "none") == ages
    assert shifted_ages(ages, "every") == pytest.approx([0.0, 0.15, 0.25, 0.45, 0.85])
    assert shifted_ages(ages, "alternate") == pytest.approx([0.0, 0.15, 0.2, 0.45, 0.8])
    with pytest.raises(ValueError):
        shifted_ages(ages, "sideways")


def test_pick_history_by_age_at_20_fps():
    times = [1.0 - 0.05 * k for k in range(20)]  # newest first, 20 FPS
    assert pick_history(times, 1.0, [0.0, 0.1, 0.2]) == [0, 2, 4]
    assert pick_history(times, 1.0, [0.0, 0.15, 0.25]) == [0, 3, 5]  # shifted: the other set of frames
    assert pick_history(times, 1.0, [5.0]) == [None]  # older than everything kept


def test_prune_keeps_half_a_step_more():
    times = [1.0 - 0.05 * k for k in range(40)]
    items = list(range(40))
    prune_history(items, times, 1.0, 0.8)
    assert times[-1] == pytest.approx(1.0 - 0.85)  # 0.8 s plus the shift, so a live switch finds its frames
    assert len(items) == len(times)
