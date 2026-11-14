import socket
import time

from mtcore.face_filter import EyeFilter, Glide
from mtservice.face import FaceStream
from mtservice.vmc import VmcSender

from test_vmc import parse


def frame(bl, br, ld=0.3, sq=0.4):
    return {"eyeBlinkLeft": bl, "eyeBlinkRight": br, "eyeLookDownLeft": ld, "eyeLookDownRight": ld,
            "eyeSquintLeft": sq, "eyeSquintRight": sq, "jawOpen": 0.2}


def run(frames, dt=0.1):
    f = EyeFilter()
    return [f.step(i * dt, x) for i, x in enumerate(frames)]


def test_open_eyes_rest_near_zero_and_blink_reaches_closed():
    # eyes open at the owner's resting levels, looking down a little; then a full blink
    out = run([frame(0.15 + 0.4 * 0.3, 0.15 + 0.4 * 0.3)] * 300 + [frame(0.72, 0.72)])
    assert out[299]["eyeBlinkLeft"] < 0.05 and out[299]["eyeSquintLeft"] < 0.05
    assert out[-1]["eyeBlinkLeft"] > 0.9 and out[-1]["eyeBlinkRight"] > 0.9
    assert out[-1]["jawOpen"] == 0.2  # other shapes pass through


def test_looking_down_does_not_close_the_eyes():
    rest = [frame(0.15 + 0.4 * ld, 0.15 + 0.4 * ld, ld=ld) for ld in [0.1, 0.6] * 300]
    out = run(rest)
    assert max(o["eyeBlinkLeft"] for o in out[-100:]) < 0.1


def test_small_asymmetry_linked_wink_kept():
    out = run([frame(0.27, 0.20)] * 200 + [frame(0.70, 0.27)])
    assert abs(out[199]["eyeBlinkLeft"] - out[199]["eyeBlinkRight"]) < 0.02
    assert out[-1]["eyeBlinkLeft"] > 0.7 and out[-1]["eyeBlinkRight"] < 0.2


def test_glide_reaches_target_before_next_frame():
    g = Glide(60)
    for k in range(5):
        g.push(k * 0.1, {"a": float(k % 2)})
        mid = g.at(k * 0.1 + 0.03)["a"]
        end = g.at(k * 0.1 + 0.09)["a"]
        if k > 0:
            assert 0.0 < mid < 1.0 and end == float(k % 2)


def test_face_stream_sends_at_its_rate():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    vmc = VmcSender("127.0.0.1", rx.getsockname()[1])
    fs = FaceStream(vmc, hz=60)
    fs.push({"jawOpen": 0.5})
    t0, n = time.monotonic(), 0
    while time.monotonic() - t0 < 0.5:
        msgs = parse(rx.recv(65536))
        assert msgs[-1][0] == "/VMC/Ext/Blend/Apply"
        n += 1
    fs.close()
    vmc.close()
    rx.close()
    assert 20 <= n <= 35  # ~30 packets in 0.5 s at 60 Hz
