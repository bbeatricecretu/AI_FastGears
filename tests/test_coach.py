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


def test_danger_alert_is_instant_and_not_said_again_by_the_llm():
    got = []
    llm = FakeLLM()
    c = Coach(llm, got.append, min_interval_s=8, summary_every_s=45)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    assert got[0]["source"] == "instant" and got[0]["severity"] == "alert"
    c.observe(st(microsleep=True, level="danger"), now=101.0)   # same episode: no repeat, no LLM echo
    time.sleep(0.2)
    assert llm.calls == 0 and len(got) == 1


def test_warning_is_worded_by_the_llm_once():
    got = []
    llm = FakeLLM()
    c = Coach(llm, got.append, min_interval_s=8, summary_every_s=1e9)
    c.observe(st(phone=True, level="caution"), now=100.0)
    time.sleep(0.3)
    assert [m["source"] for m in got] == ["fake"]                 # one message, in the LLM's words
    assert got[0]["severity"] == "warning" and got[0]["reason"] == "phone in use"


class SlowLLM(FakeLLM):
    def complete(self, system, user, json_mode=False):
        time.sleep(3.2)
        return super().complete(system, user, json_mode)


def test_slow_llm_warning_falls_back_to_rule_text_and_late_reply_is_dropped():
    got = []
    c = Coach(SlowLLM(), got.append, min_interval_s=8, summary_every_s=1e9)
    c.observe(st(phone=True, level="caution"), now=100.0)
    time.sleep(2.8)
    assert [m["source"] for m in got] == ["instant"]              # rule text after WARN_LLM_WAIT_S
    time.sleep(0.8)
    assert len(got) == 1                                          # the late LLM wording is not said too


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


def test_checkin_no_answer_but_eyes_on_road_passes_quietly():
    ck = CheckIn(answer_timeout_s=10, max_rounds=3, cooldown_s=0)
    ck.start("microsleep", 0.0)
    job = None
    for i in range(1, 12):
        job = job or ck.observe(road(), float(i))
    assert job and job["answer"] is None                           # timed out, but awake
    assert ck.resolve(job, merge_verdict(rule_verdict(job), None), 11.0) == [] and not ck.active


def test_checkin_asks_again_only_while_camera_says_unsafe_then_stops():
    ck = CheckIn(answer_timeout_s=10, max_rounds=2, cooldown_s=0)
    first = ck.start("microsleep", 0.0)
    job = None
    for i in range(1, 12):
        job = job or ck.observe(st(face_present=True, looking="down", eyes_closed=False), float(i))
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 11.0)
    assert out[0]["expects_answer"] and ck.round == 2
    assert out[0]["message"] != first["message"]                   # a new question
    ck.observe(st(face_present=True, looking="road", level="danger"), 12.0)   # still in danger
    job = ck.answer("uh", 12.5)
    assert job["reaction"]["level_now"] == "danger"
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 13.0)
    assert out[0]["reason"] == "check-in failed" and out[0]["severity"] == "alert" and not ck.active


def test_no_other_voice_messages_during_a_checkin():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, checkin=CheckIn(answer_timeout_s=60, cooldown_s=30))
    c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)
    n = len(got)                                                   # alert + question
    c.observe(st(distracted=True, looking="left", level="danger"), now=125.0)   # same hazard, past repeat
    c.observe(st(hand_off_wheel=True, level="caution"), now=126.0)             # a mere warning
    assert len(got) == n
    c.observe(st(microsleep=True, level="danger"), now=127.0)                  # a new danger interrupts
    assert got[-1]["reason"] == "microsleep"


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


# ── conversation ────────────────────────────────────────────────────────────────

class EchoLLM:
    name = "fake"
    def __init__(self): self.users = []
    def complete(self, system, user, json_mode=False):
        self.users.append(user)
        return "Sounds good."


def test_conversation_remembers_previous_turns_and_keeps_the_mic_open():
    llm = EchoLLM()
    c = Coach(llm, lambda m: None, min_interval_s=1e9, summary_every_s=1e9)
    first = c.chat("I'm driving to Cluj", st())
    assert first["expects_answer"]                                  # browser reopens the mic
    c.chat("about two hours left", st())
    assert "driver: I'm driving to Cluj" in llm.users[1] and "co-pilot: Sounds good." in llm.users[1]


def test_tiredness_opens_a_conversation_instead_of_a_plain_warning():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9)
    c.observe(st(tired=True, level="caution"), now=10.0)
    assert got[0]["expects_answer"] and got[0]["reason"] == "signs of fatigue"


def test_calm_drive_gets_an_occasional_conversation_opener():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, convo_idle_s=600)
    c.observe(st(), now=0.0)
    c.observe(st(), now=599.0)
    assert not got
    c.observe(st(), now=601.0)
    assert got[-1]["reason"] == "conversation" and got[-1]["expects_answer"]
    c.chat("all fine", st())                                       # talking resets the idle timer
    c.observe(st(), now=900.0)
    assert sum(m.get("reason") == "conversation" for m in got) == 1


# ── message pacing ──────────────────────────────────────────────────────────────

def test_warnings_are_spaced_by_message_gap():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, message_gap_s=12)
    c.observe(st(phone=True, level="caution"), now=100.0)                 # warning spoken
    c.observe(st(tired=True, level="caution"), now=103.0)                 # other warning: too soon
    assert [m["reason"] for m in got] == ["phone in use"]
    c.observe(st(tired=True, level="caution"), now=113.0)                 # gap over: now it is said
    assert [m["reason"] for m in got] == ["phone in use", "signs of fatigue"]


def test_danger_alert_ignores_message_gap():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, message_gap_s=60)
    c.observe(st(phone=True, level="caution"), now=100.0)
    c.observe(st(microsleep=True, level="danger"), now=101.0)
    assert [m["severity"] for m in got] == ["warning", "alert"]


def test_flickering_warning_needs_hold_time():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, warn_hold_s=1.5)
    c.observe(st(phone=True, level="caution"), now=100.0)
    c.observe(st(), now=100.5)
    assert got == []                                                      # a blip stays silent
    c.observe(st(phone=True, level="caution"), now=101.0)
    c.observe(st(phone=True, level="caution"), now=101.6)                 # same episode, held 1.6 s
    assert [m["reason"] for m in got] == ["phone in use"]


def test_presets_and_custom_tuning():
    c = Coach(None, lambda m: None, checkin=CheckIn())
    t = c.set_tuning("quiet")
    assert t["preset"] == "quiet" and t["values"]["message_gap_s"] == 25 and c.checkin.cooldown_s == 180
    t = c.set_tuning(message_gap_s=999, questions=False)                  # clamped, now custom
    assert t["preset"] == "custom" and t["values"]["message_gap_s"] == 120 and not t["values"]["questions"]
    assert c.set_tuning("normal")["preset"] == "normal"


def test_questions_can_be_switched_off():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, checkin=CheckIn())
    c.set_tuning(questions=False)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    assert [m["source"] for m in got] == ["instant"] and not c.checkin.active


# ── waiting for the driver's reply: no message jam ──────────────────────────────

def test_open_question_silences_everything_but_microsleep():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, convo_idle_s=600)
    c.observe(st(), now=0.0)
    c.observe(st(), now=601.0)                                         # opener: a question
    assert got[-1]["expects_answer"] and c.awaiting_reply(601.0)
    n = len(got)
    c.observe(st(phone=True, level="caution"), now=603.0)              # warning: waits
    c.observe(st(distracted=True, looking="left", level="danger"), now=605.0)   # other danger: waits too
    assert len(got) == n
    c.observe(st(microsleep=True, level="danger"), now=606.0)          # can't answer asleep: interrupts
    assert got[-1]["reason"] == "microsleep"


def test_reply_closes_the_question_and_talk_resumes():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9, convo_idle_s=600)
    c.observe(st(), now=0.0)
    c.observe(st(), now=601.0)
    c.observe(st(), now=602.0)
    c.chat("all fine, heading home", st())                             # answer -> co-pilot reply (asks again)
    assert got[-1]["reason"] == "chat"
    c.observe(st(), now=630.0)                                         # unanswered follow-up expired
    assert not c.awaiting_reply(630.0)
    c.observe(st(phone=True, level="caution"), now=631.0)
    assert got[-1]["reason"] == "phone in use"


def test_driver_talking_keeps_the_floor_and_the_checkin_open():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9,
              checkin=CheckIn(answer_timeout_s=12, cooldown_s=30))
    c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)
    assert got[-1]["reason"] == "check-in"
    c.observe(road(), now=105.0)
    c.set_answering(True)                                              # browser: driver is speaking
    for t in range(106, 125):                                          # past the 12 s answer window
        c.observe(road(), now=float(t))
    assert c.checkin.waiting and got[-1]["reason"] == "check-in"       # no timeout, nothing said
    assert c.chat("a white van", road())["source"] == "check"          # the late answer still counts


def test_llm_comment_arriving_during_a_question_is_dropped():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9)
    c._t = 10.0
    c._emit({"message": "Where are you heading?", "severity": "info", "reason": "conversation",
             "source": "instant", "expects_answer": True})
    c._submit(("react", {}, {"message": "Phone away.", "severity": "warning", "reason": "phone in use"}))
    time.sleep(0.2)
    assert [m["message"] for m in got] == ["Where are you heading?"]


# ── ongoing danger after a passed check-in: no message jam ─────────────────────

def _off(level="danger"):
    return st(distracted=True, looking="left", level=level)


def test_danger_flicker_is_one_message_not_a_stream():
    llm = FakeLLM()
    got = []
    c = Coach(llm, got.append, min_interval_s=8, summary_every_s=90, message_gap_s=0)
    t = 100.0
    for i in range(90):                                    # 18 s of danger flickering with caution
        c.observe(_off("caution" if i % 10 >= 6 else "danger"), now=t)
        t += 0.2
        time.sleep(0.002)
    time.sleep(0.2)
    assert llm.calls == 0 and len(got) == 1                # the alert once: no LLM echo, no step-down warning


def test_ongoing_danger_gets_a_new_question_instead_of_repeated_alerts():
    got = []
    c = Coach(None, got.append, min_interval_s=1e9, summary_every_s=1e9,
              checkin=CheckIn(answer_timeout_s=20, cooldown_s=90))
    c.observe(_off(), now=5.0)                             # alert + question
    c.observe(road(), now=8.0)
    c.chat("a car ahead", road())                          # passes
    time.sleep(0.2)
    t = 11.0
    while t < 50.0:                                        # looks away again, inside the 90 s cooldown
        c.observe(_off(), now=t)
        t += 0.2
    questions = [m for m in got if m.get("reason") == "check-in"]
    assert len(questions) == 2                             # asked again once the alert repeated
    assert sum(m.get("source") == "instant" for m in got) <= 3
