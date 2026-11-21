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


def test_pattern_explanation_validation_cache_and_fallback():
    import json
    from fastgears.llm.coach import PatternReply
    sample=dict(pattern_name="Repeated distraction", history=["LOOKING_LEFT"], severity="CAUTION",
                evidence=[dict(id="a",type="LOOKING_LEFT",duration=.6)])
    valid=PatternReply(pattern_name=sample["pattern_name"], severity_suggestion="WARNING", message="Repeated glances.",
                       voice_message="Eyes on the road.", reason="Separate glances", recommended_action="Look forward.",
                       evidence_ids=["a"])
    class Structured:
        name="gemini"
        calls=0
        response=valid.model_dump_json()
        def complete(self, system, prompt, json_mode=False):
            self.calls+=1
            assert json_mode and json.loads(prompt)==sample
            return self.response
    client=Structured()
    got, ready=[],threading.Event()
    coach=Coach(client,lambda m:(got.append(m),ready.set()))
    try:
        assert coach._pattern_reply(sample)["severity"]=="info"
        assert coach._pattern_reply(sample)["voice_message"]=="Eyes on the road."
        assert client.calls==1
        coach.cache.clear()
        client.response=valid.model_copy(update={"evidence_ids":["unseen"]}).model_dump_json()
        coach._submit(("pattern",sample,dict(message="Refocus.",severity="info",reason="local",alert_id="abc")))
        assert ready.wait(2)
        assert got[-1]["source"]=="rules" and coach.service_status=="invalid response"
    finally:
        coach.stop()


def test_missing_key_failure_backoff_and_slow_provider():
    from types import SimpleNamespace
    from fastgears.llm.coach import make_client
    assert make_client(SimpleNamespace(llm_provider="gemini",gemini_api_key=None)) is None
    gate, entered, emitted = threading.Event(), threading.Event(), threading.Event()
    class Slow:
        name="gemini"
        calls=0
        def complete(self,*args,**kwargs):
            self.calls+=1
            entered.set()
            gate.wait(3)
            raise TimeoutError()
    client=Slow()
    messages=[]
    def deliver(m):
        messages.append(m)
        if m["source"]=="rules":
            emitted.set()
    coach=Coach(client,deliver)
    try:
        coach.observe(st(level="danger",microsleep=True),now=100)
        assert entered.wait(1) and messages[0]["severity"]=="alert"
        gate.set()
        assert emitted.wait(2) and coach.service_status.startswith("unavailable")
        coach.chat("Status?",st())
        assert client.calls==1
    finally:
        gate.set()
        coach.stop()
