import threading
import time

from fastgears.llm.coach import Coach, compact_state, local_message, parse_reply


def st(**a):
    base = {"calibrated": True, "level": "ok", "reasons": []}
    return {"attention": {**base, **a}, "pulse": {"bpm": 70, "trust": {"label": "unreliable"}}, "uptime_s": 120}


def test_local_message_priorities():
    assert local_message(st()) is None
    assert local_message(st(microsleep=True, phone=True))["reason"] == "microsleep"
    assert local_message(st(phone=True))["severity"] == "warning"               # just seen
    assert local_message(st(phone=True, level="danger"))["severity"] == "alert"  # 10 s of use
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
        on = i % 2 == 0
        c.observe(st(phone=on, level="danger" if on else "ok"), now=100.0 + i)
    assert sum(m.get("source") == "instant" for m in got) == 1       # one alert for the whole episode


def test_every_danger_is_spoken_instantly():
    """Regression: only microsleep / phone / head drop used to get an instant voice alert."""
    cases = [dict(distracted=True, looking="left", level="danger"),
             dict(eyes_closing=True, level="danger"),
             dict(hand_off_wheel=True, level="danger"),
             dict(tired=True, level="danger"),
             dict(level="danger")]                                  # danger with no specific rule
    for a in cases:
        got = []
        c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9)
        c.observe(st(**a), now=100.0)
        assert got and got[0]["source"] == "instant" and got[0]["severity"] == "alert", a


def test_warning_spoken_once_then_escalation_spoken_again():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9)
    for i in range(10):                                             # phone seen: warning, once
        c.observe(st(phone=True, level="caution"), now=100.0 + i)
    assert [m["severity"] for m in got] == ["warning"]
    c.observe(st(phone=True, level="danger"), now=111.0)          # 10 s of use: danger
    assert [m["severity"] for m in got] == ["warning", "alert"]


# ── closed-loop check-in ────────────────────────────────────────────────────────

from fastgears.llm.intervention import CheckIn, merge_verdict, rule_verdict  # noqa: E402


def road(**a):
    return st(face_present=True, looking="road", eyes_closed=False, **a)


def test_checkin_passes_with_answer_and_eyes_on_road():
    ck = CheckIn(answer_timeout_s=12, max_rounds=3, cooldown_s=30)
    q = ck.start("eyes off road", 100.0)
    assert q["expects_answer"] and ck.waiting
    for i in range(1, 6):
        assert ck.observe(road(), 100.0 + i) is None
    job = ck.answer("a white van", 105.5)
    assert job["reaction"]["on_road_share"] == 1.0 and job["answer_latency_s"] == 5.5
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 106.0)
    assert out[0]["reason"] == "check-in passed" and not ck.active
    assert ck.start("phone in use", 120.0) is None                 # cooldown
    assert ck.start("phone in use", 136.0) is not None


def test_checkin_no_answer_asks_again_then_stops():
    ck = CheckIn(answer_timeout_s=10, max_rounds=2, cooldown_s=0)
    first = ck.start("microsleep", 0.0)
    job = None
    for i in range(1, 12):
        job = job or ck.observe(road(), float(i))
    assert job and job["answer"] is None                           # timed out
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 11.0)
    assert out[0]["expects_answer"] and ck.round == 2
    assert out[0]["message"] != first["message"]                   # a new question
    ck.observe(st(face_present=True, looking="road", level="danger"), 12.0)   # still in danger
    job = ck.answer("uh", 12.5)
    assert job["reaction"]["level_now"] == "danger"
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 13.0)
    assert out[0]["reason"] == "check-in failed" and out[0]["severity"] == "alert" and not ck.active


def test_checkin_eyes_off_road_fails_even_with_answer():
    ck = CheckIn()
    ck.start("eyes off road", 0.0)
    for i in range(1, 8):
        ck.observe(st(face_present=True, looking="down", eyes_closed=False), float(i))
    job = ck.answer("a red car", 8.0)
    assert job["reaction"]["on_road_share"] == 0.0
    assert rule_verdict(job)["ok"] is False


def test_llm_cannot_pass_a_driver_the_camera_fails():
    job = {"answer": "the road", "reaction": {"level_now": "danger", "on_road_share": 0.2, "microsleep": False}}
    v = merge_verdict(rule_verdict(job), {"ok": True, "message": "Great!", "next_question": ""})
    assert v["ok"] is False
    v = merge_verdict({**rule_verdict(job), "ok": True}, {"ok": False, "message": "Say what you see.",
                                                          "next_question": "What is ahead?"})
    assert v["ok"] is False and v["next_question"] == "What is ahead?"


class JudgeLLM:
    name = "fake"
    def __init__(self): self.judged = 0
    def complete(self, system, user, json_mode=False):
        if "alertness check" in system:
            self.judged += 1
            return '{"ok": true, "message": "Good, stay focused.", "next_question": ""}'
        return '{"message":"Careful.","severity":"warning","reason":"x"}'


def test_coach_runs_the_closed_loop():
    got, done = [], threading.Event()
    def on(m):
        got.append(m)
        if m.get("reason") == "check-in passed":
            done.set()
    llm = JudgeLLM()
    c = Coach(llm, on, min_interval_s=1e9, summary_every_s=1e9, checkin=CheckIn(cooldown_s=30))
    c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)
    assert got[0]["source"] == "instant" and got[1]["expects_answer"]   # alert, then the question
    assert c.checkin_status()["active"] and c.checkin_status()["round"] == 1
    for i in range(1, 5):
        c.observe(road(), now=100.0 + i)
    assert c.chat("a truck ahead", road())["source"] == "check"      # the answer, not a chat
    assert done.wait(2) and llm.judged == 1
    assert got[-1]["message"] == "Good, stay focused." and not c.checkin_status()["active"]


def test_local_message_head_dropping():
    from fastgears.llm.coach import local_message
    m = local_message({"attention": {"looking": "nodding", "level": "danger",
                                     "reasons": ["head dropping 3.0 s (nodding off?)"]}})
    assert m["reason"] == "head tilted" and m["severity"] == "alert"


def test_compact_state_names_one_primary_hazard():
    c = compact_state(st(level="danger", phone=True, perclos=0.2, yawns_5min=2))
    assert c["primary_hazard"] == "phone in use"
    assert compact_state(st(level="ok"))["primary_hazard"] is None
