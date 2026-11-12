from mtservice.detector import unmirror_blendshapes


def test_unmirror_swaps_left_and_right_only():
    out = unmirror_blendshapes({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.9, "mouthSmileLeft": 0.3, "jawOpen": 0.5})
    assert out == {"eyeBlinkRight": 0.1, "eyeBlinkLeft": 0.9, "mouthSmileRight": 0.3, "jawOpen": 0.5}
