"""Regression tests for bugs found in the bug burn-down (see BUG_BURN_DOWN.md)."""
import threading
import time

import fastgears.llm.coach as coach_mod
from fastgears.llm.coach import Coach
from fastgears.llm.intervention import CheckIn


def st(**a):
    base = {"calibrated": True, "level": "ok", "reasons": []}
    return {"attention": {**base, **a}, "pulse": {}, "uptime_s": 120}


class HangingLLM:
    """An LLM whose network call never returns (stalled connection)."""
    name = "hang"

    def __init__(self):
        self.release = threading.Event()

    def complete(self, system, user, json_mode=False):
        self.release.wait(30)
        return '{"action":"continue","message":"ok"}'


def test_bug1_hanging_judge_llm_does_not_mute_later_danger_alerts(monkeypatch):
    monkeypatch.setattr(coach_mod, "JUDGE_WAIT_S", 0.3)
    got = []
    llm = HangingLLM()
    c = Coach(llm, got.append, checkin=CheckIn(answer_timeout_s=60, cooldown_s=30))
    try:
        c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)   # alert + question
        assert any(m.get("expects_answer") for m in got)
        for t in (102.5, 103.0, 104.0):                  # eyes back on the road
            c.observe(st(looking="road", face_present=True), now=t)
        c.chat("a red car", st(looking="road"))          # the answer goes to the (hanging) LLM judge
        time.sleep(0.8)                                  # judge wait elapsed: the rules decide
        c.observe(st(looking="road"), now=105.0)
        assert not c.checkin.active                      # check-in no longer stuck in "judging"
        n = len(got)
        c.observe(st(phone=True, level="danger"), now=140.0)   # a new danger later on
        assert len(got) > n and got[n]["severity"] == "alert"
    finally:
        llm.release.set()


def _timeout_ms(genai_client):
    return genai_client._api_client._http_options.timeout


def test_bug2_gemini_clients_have_a_network_timeout():
    from fastgears.llm.coach import GeminiClient
    from fastgears.vision.phone_check import VisionPhoneCheck
    assert _timeout_ms(GeminiClient("test-key", "m")._client) == 15000
    assert _timeout_ms(VisionPhoneCheck("test-key", "m")._client) == 15000


def test_bug3_numpy_value_in_state_does_not_break_the_broadcast():
    import asyncio

    import numpy as np

    from fastgears.server import Hub

    class WS:
        def __init__(self): self.sent = []
        async def send_text(self, d): self.sent.append(d)

    h, ws = Hub(), WS()
    h.clients.add(ws)
    asyncio.run(h.broadcast({"type": "state", "x": np.float32(1.5), "b": np.array([1, 2])}))
    assert ws.sent and '"x": 1.5' in ws.sent[0] and '"b": [1, 2]' in ws.sent[0]


def test_bug4_no_camera_state_drops_last_frames_detections():
    from fastgears.pipeline import Pipeline
    p = Pipeline.__new__(Pipeline)
    p.rppg = p.rgb = None
    p.s = type("S", (), {"enable_rgb": False})()
    p.camera = type("C", (), {"connected": False})()
    from fastgears.vision.discomfort import DiscomfortMonitor
    p.discomfort = DiscomfortMonitor()
    import threading
    p._lock = threading.Lock()
    p._state = {"objects": [{"label": "cell phone", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
                "hands": {"available": True, "seen": 2, "driver": 2, "free": 1, "phone_source": "yolo"},
                "attention": {"calibrated": True}}
    p._publish_no_camera()
    s = p.state()
    assert s["objects"] == [] and s["hands"]["seen"] == 0 and s["hands"]["available"] is True
    assert s["status"] == "no camera"
