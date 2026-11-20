import threading
import time

from fastgears.llm.coach import Coach, compact_state, local_message, parse_reply


def st(**a):
    base = {"calibrated": True, "level": "ok", "reasons": []}
    return {"attention": {**base, **a}, "pulse": {"bpm": 70, "trust": {"label": "unreliable"}}, "uptime_s": 120}


def test_local_message_priorities():
    assert local_message(st()) is None
    assert local_message(st(microsleep=True, phone=True))["reason"] == "microsleep"
    assert local_message(st(phone=True))["severity"] == "alert"
    assert "left" in local_message(st(distracted=True, looking="left"))["message"]


def test_compact_state_drops_unreliable_pulse():
    c = compact_state(st(level="caution"))
    assert "pulse_bpm" not in c and c["level"] == "caution" and c["driving_minutes"] == 2.0


def test_parse_reply():
    assert parse_reply('```json\n{"message":"Eyes up","severity":"warning","reason":"x"}\n```')["severity"] == "warning"
    assert parse_reply('{"message":"hi","severity":"weird"}')["severity"] == "info"
    assert parse_reply("") is None


class FakeLLM:
    name = "fake"
    def __init__(self): self.calls = 0
    def complete(self, system, user, json_mode=False):
        self.calls += 1
        return '{"message":"Take a break soon.","severity":"warning","reason":"fatigue"}'


def test_coach_instant_alert_then_llm():
    got, ev = [], threading.Event()
    def on(m):
        got.append(m)
        if m["source"] == "fake":
            ev.set()
    llm = FakeLLM()
    c = Coach(llm, on, min_interval_s=8, summary_every_s=45)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    assert got[0]["source"] == "instant" and got[0]["severity"] == "alert"
    assert ev.wait(2)
    c.observe(st(microsleep=True, level="danger"), now=101.0)   # same reason: no new LLM call, no repeat alert
    time.sleep(0.1)
    assert llm.calls == 1 and sum(m["source"] == "instant" for m in got) == 1


def test_coach_without_llm_uses_rules():
    got, ev = [], threading.Event()
    c = Coach(None, lambda m: (got.append(m), ev.set()), summary_every_s=0)
    c.observe(st(), now=1.0)
    assert ev.wait(2) and got[0]["source"] == "rules"
    assert "offline" in c.chat("how am I?", st())["message"]


def test_flickering_phone_alerts_once():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9)
    for i in range(20):                      # phone on/off every second for 20 s
        c.observe(st(phone=(i % 2 == 0), level="danger"), now=100.0 + i)
    assert sum(m.get("source") == "instant" for m in got) == 1       # one alert for the whole episode
