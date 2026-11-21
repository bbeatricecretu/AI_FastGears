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
    c = Coach(llm, got.append)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    assert got[0]["source"] == "instant" and got[0]["severity"] == "alert"
    c.observe(st(microsleep=True, level="danger"), now=101.0)   # same episode: no repeat, no LLM echo
    time.sleep(0.2)
    assert llm.calls == 0 and len(got) == 1


def test_warning_is_worded_by_the_llm_once():
    got = []
    llm = FakeLLM()
    c = Coach(llm, got.append)
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
    c = Coach(SlowLLM(), got.append)
    c.observe(st(phone=True, level="caution"), now=100.0)
    time.sleep(2.8)
    assert [m["source"] for m in got] == ["instant"]              # rule text after WARN_LLM_WAIT_S
    time.sleep(0.8)
    assert len(got) == 1                                          # the late LLM wording is not said too


def test_coach_without_llm_uses_rules():
    got = []
    c = Coach(None, got.append)
    c.observe(st(phone=True, level="caution"), now=1.0)
    assert got[0]["source"] == "instant"
    assert "offline" in c.chat("how am I?", st())["message"]


def test_flickering_phone_alerts_once():
    got = []
    c = Coach(None, got.append)
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
        c = Coach(None, got.append)
        c.observe(st(**a), now=100.0)
        assert got and got[0]["source"] == "instant" and got[0]["severity"] == "alert", a


def test_warning_spoken_once_then_escalation_spoken_again():
    got = []
    c = Coach(None, got.append)
    for i in range(10):                                             # phone seen: warning, once
        c.observe(st(phone=True, level="caution"), now=100.0 + i)
    assert [m["severity"] for m in got] == ["warning"]
    c.observe(st(phone=True, level="danger"), now=111.0)          # 10 s of use: danger
    assert [m["severity"] for m in got] == ["warning", "alert"]


# ── closed-loop check-in ────────────────────────────────────────────────────────

from fastgears.llm.intervention import PASS_MSG, CheckIn, merge_verdict, rule_verdict  # noqa: E402


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
    ck.start("eyes off road", 0.0)
    job = None
    for i in range(1, 12):
        job = job or ck.observe(road(), float(i))
    assert job and job["answer"] is None                           # timed out, but awake
    assert ck.resolve(job, merge_verdict(rule_verdict(job), None), 11.0) == [] and not ck.active
    ck.start("microsleep", 20.0)                                   # after a drowsy alert: no silent pass,
    job = None                                                     # the break recommendation is said
    for i in range(21, 32):
        job = job or ck.observe(road(), float(i))
    out = ck.resolve(job, merge_verdict(rule_verdict(job), None), 31.0)
    assert [m["reason"] for m in out] == ["take a break"] and "Good answer" not in out[0]["message"]


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
    c = Coach(None, got.append, checkin=CheckIn(answer_timeout_s=60, cooldown_s=30))
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
    c = Coach(llm, on, checkin=CheckIn(cooldown_s=30))
    c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)
    assert got[0]["source"] == "instant" and got[1]["expects_answer"]   # alert, then the question
    assert c.checkin_status()["active"] and c.checkin_status()["round"] == 1
    for i in range(1, 5):
        c.observe(road(), now=100.0 + i)
    assert c.chat("a truck ahead", road())["source"] == "check"      # the answer, not a chat
    assert done.wait(2) and llm.judged == 1
    assert got[-1]["message"] == PASS_MSG and not c.checkin_status()["active"]   # fixed thanks, not the LLM's


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
    c = Coach(llm, lambda m: None)
    first = c.chat("I'm driving to Cluj", st())
    assert first["expects_answer"]                                  # browser reopens the mic
    c.chat("about two hours left", st())
    assert "driver: I'm driving to Cluj" in llm.users[1] and "co-pilot: Sounds good." in llm.users[1]


def test_tiredness_opens_a_conversation_instead_of_a_plain_warning():
    got = []
    c = Coach(None, got.append)
    c.observe(st(tired=True, level="caution"), now=10.0)
    assert got[0]["expects_answer"] and got[0]["reason"] == "signs of fatigue"


def test_calm_drive_stays_silent():
    """No summaries, praise or small talk: without a hazard nothing is said, even for a long time."""
    got = []
    c = Coach(FakeLLM(), got.append)
    for i in range(0, 3600, 5):
        c.observe(st(), now=float(i))
    time.sleep(0.1)
    assert got == []


def test_no_praise_when_a_danger_flickers_to_ok():
    """Regression (demo video): "You are doing great" between "head dropping" alerts."""
    llm = FakeLLM()
    got = []
    c = Coach(llm, got.append)
    for i in range(300):                                  # 60 s: head dropping, with ok frames in between
        drop = i % 7 != 0
        c.observe(st(looking="tilted", reasons=["head tilted"], level="danger") if drop else st(), now=100 + i * 0.2)
    time.sleep(0.1)
    assert llm.calls == 0
    assert all(m["reason"] == "head tilted" for m in got)


# ── message pacing ──────────────────────────────────────────────────────────────

def test_warnings_are_spaced_by_message_gap():
    got = []
    c = Coach(None, got.append, message_gap_s=12)
    c.observe(st(phone=True, level="caution"), now=100.0)                 # warning spoken
    c.observe(st(tired=True, level="caution"), now=103.0)                 # other warning: too soon
    assert [m["reason"] for m in got] == ["phone in use"]
    c.observe(st(tired=True, level="caution"), now=113.0)                 # gap over: now it is said
    assert [m["reason"] for m in got] == ["phone in use", "signs of fatigue"]


def test_danger_alert_ignores_message_gap():
    got = []
    c = Coach(None, got.append, message_gap_s=60)
    c.observe(st(phone=True, level="caution"), now=100.0)
    c.observe(st(microsleep=True, level="danger"), now=101.0)
    assert [m["severity"] for m in got] == ["warning", "alert"]


def test_flickering_warning_needs_hold_time():
    got = []
    c = Coach(None, got.append, warn_hold_s=1.5)
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
    c = Coach(None, got.append, checkin=CheckIn())
    c.set_tuning(questions=False)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    assert [m["source"] for m in got] == ["instant"] and not c.checkin.active


# ── waiting for the driver's reply: no message jam ──────────────────────────────

def test_open_question_silences_everything_but_microsleep():
    got = []
    c = Coach(None, got.append)
    c.observe(st(tired=True, level="caution"), now=601.0)              # fatigue opener: a question
    assert got[-1]["expects_answer"] and c.awaiting_reply(601.0)
    n = len(got)
    c.observe(st(phone=True, level="caution"), now=603.0)              # warning: waits
    c.observe(st(distracted=True, looking="left", level="danger"), now=605.0)   # other danger: waits too
    assert len(got) == n
    c.observe(st(microsleep=True, level="danger"), now=606.0)          # can't answer asleep: interrupts
    assert got[-1]["reason"] == "microsleep"


def test_reply_closes_the_question_and_talk_resumes():
    got = []
    c = Coach(None, got.append)
    c.observe(st(tired=True, level="caution"), now=601.0)              # fatigue opener: a question
    c.observe(st(), now=602.0)
    c.chat("all fine, heading home", st())                             # answer -> co-pilot reply (asks again)
    assert got[-1]["reason"] == "chat"
    c.observe(st(), now=630.0)                                         # unanswered follow-up expired
    assert not c.awaiting_reply(630.0)
    c.observe(st(phone=True, level="caution"), now=631.0)
    assert got[-1]["reason"] == "phone in use"


def test_driver_talking_keeps_the_floor_and_the_checkin_open():
    got = []
    c = Coach(None, got.append, checkin=CheckIn(answer_timeout_s=12, cooldown_s=30))
    c.observe(st(distracted=True, looking="left", level="danger"), now=100.0)
    assert got[-1]["reason"] == "check-in"
    c.observe(road(), now=105.0)
    c.set_answering(True)                                              # browser: driver is speaking
    for t in range(106, 125):                                          # past the 12 s answer window
        c.observe(road(), now=float(t))
    assert c.checkin.waiting and got[-1]["reason"] == "check-in"       # no timeout, nothing said
    assert c.chat("a white van", road())["source"] == "check"          # the late answer still counts


# ── ongoing danger after a passed check-in: no message jam ─────────────────────

def _off(level="danger"):
    return st(distracted=True, looking="left", level=level)


def test_danger_flicker_is_one_message_not_a_stream():
    llm = FakeLLM()
    got = []
    c = Coach(llm, got.append, message_gap_s=0)
    t = 100.0
    for i in range(90):                                    # 18 s of danger flickering with caution
        c.observe(_off("caution" if i % 10 >= 6 else "danger"), now=t)
        t += 0.2
        time.sleep(0.002)
    time.sleep(0.2)
    assert llm.calls == 0 and len(got) == 1                # the alert once: no LLM echo, no step-down warning


def test_ongoing_danger_gets_a_new_question_instead_of_repeated_alerts():
    got = []
    c = Coach(None, got.append, checkin=CheckIn(answer_timeout_s=20, cooldown_s=90))
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


def _answer_round(ck, t, answer):
    for i in range(1, 5):
        ck.observe(road(), t + i)
    return ck.answer(answer, t + 5)


def test_round_one_alternates_road_and_check_questions():
    from fastgears.llm.intervention import CHECK_QUESTIONS, QUESTIONS
    ck = CheckIn(cooldown_s=0)
    kinds = []
    for n in range(4):
        q = ck.start("eyes off road", n * 100.0)
        kinds.append(ck.kind)
        assert q["message"] in (QUESTIONS if ck.kind == "road" else [c[0] for c in CHECK_QUESTIONS])
        job = _answer_round(ck, n * 100.0, "12 a red car")
        ck.resolve(job, {"ok": True, "action": "continue", "message": ""}, n * 100.0 + 6)
    assert kinds == ["road", "check", "road", "check"]


def test_wrong_check_answer_fails_right_one_passes_without_llm():
    ck = CheckIn(cooldown_s=0)
    ck.start("eyes off road", 0.0)                                 # check-in 1: road, anything goes
    job = _answer_round(ck, 0.0, "a purple elephant")
    assert rule_verdict(job)["ok"]
    ck.resolve(job, merge_verdict(rule_verdict(job), None), 6.0)
    ck.start("eyes off road", 10.0)                                # check-in 2: "What is seven plus five?"
    assert ck.kind == "check" and ck.expected == ("12", "twelve")
    job = _answer_round(ck, 10.0, "eleven")
    v = merge_verdict(rule_verdict(job), None)
    assert not v["ok"] and v["message"] == "That's not right. Stay with me."
    out = ck.resolve(job, v, 16.0)
    assert out[0]["expects_answer"] and ck.round == 2 and ck.kind == "check"   # follow-up is a check question
    job = _answer_round(ck, 20.0, "it's seven")                    # "What is three plus four?"
    assert merge_verdict(rule_verdict(job), None)["ok"]


def test_llm_judges_check_answers_in_any_wording_but_camera_floor_stays():
    from fastgears.llm.intervention import evidence, parse_judge
    job = {"kind": "check", "expected": ["12", "twelve"], "question": "What is seven plus five?",
           "answer": "doisprezece", "round": 1, "max_rounds": 3,
           "reaction": {"level_now": "ok", "on_road_share": 1.0, "microsleep": False}}
    e = evidence(job)
    assert e["question_type"] == "check" and e["correct_answer"] == "12"
    assert not rule_verdict(job)["ok"]                               # word matching misses it ...
    llm = parse_judge('{"action": "continue", "message": "Correct, thanks."}')
    assert merge_verdict(rule_verdict(job), llm)["ok"]              # ... the LLM accepts it
    job["reaction"]["on_road_share"] = 0.1                           # eyes off the road: still a fail
    assert not merge_verdict(rule_verdict(job), llm)["ok"]


# ── demo-video regressions ─────────────────────────────────────────────────────

def test_one_drowsy_episode_is_not_a_burst_of_alerts():
    """Eyes closing, head dropping, generic danger... one after another used to be one alert each."""
    got = []
    c = Coach(None, got.append)
    seq = [dict(eyes_closing=True, level="danger"),
           dict(looking="tilted", reasons=["head tilted"], level="danger"),
           dict(level="danger"),
           dict(hand_off_wheel=True, level="danger")]
    t = 100.0
    for i in range(50):                                   # 10 s, the cause changes every 0.6 s
        c.observe(st(**seq[(i // 3) % len(seq)]), now=t)
        t += 0.2
    assert len(got) == 1


def test_microsleep_always_interrupts_the_alert_pace():
    got = []
    c = Coach(None, got.append)
    c.observe(st(eyes_closing=True, level="danger"), now=100.0)
    c.observe(st(microsleep=True, level="danger"), now=101.0)
    assert [m["reason"] for m in got] == ["eyes closing", "microsleep"]


def test_warning_waits_after_an_alert():
    got = []
    c = Coach(None, got.append, message_gap_s=12)
    c.observe(st(microsleep=True, level="danger"), now=100.0)
    c.observe(st(phone=True, level="caution"), now=103.0)
    assert [m["reason"] for m in got] == ["microsleep"]


def test_eyes_off_road_wording_is_grammatical():
    for where in ("nodding", "tilted", "away", "road"):
        m = local_message(st(distracted=True, looking=where))["message"]
        assert "Stop looking" not in m and m.endswith("Look ahead."), m
    assert "Stop looking up" in local_message(st(distracted=True, looking="up"))["message"]


def test_drowsy_checkin_passes_only_with_a_break_never_alert():
    job = {"answer": "twelve", "kind": "check", "expected": ["12", "twelve"], "hazard": "head tilted",
           "reply_time_s": 1.0, "reaction": {"on_road_share": 1.0, "level_now": "ok", "microsleep": False,
                                              "drowsiness_avg": 10.0, "distraction_avg": 0.0}}
    v = rule_verdict(job)
    assert v["ok"] and v["action"] == "break"
    llm = {"ok": True, "action": "continue", "message": "Good response. You are alert.", "next_question": ""}
    m = merge_verdict(v, llm)
    assert m["action"] == "break" and "alert" not in m["message"].lower()


def test_llm_pass_message_is_the_fixed_thanks():
    from fastgears.llm.intervention import PASS_MSG
    job = {"answer": "a red car", "kind": "road", "hazard": "eyes off road", "reply_time_s": 1.0,
           "reaction": {"on_road_share": 1.0, "level_now": "ok", "microsleep": False,
                        "drowsiness_avg": 0.0, "distraction_avg": 0.0}}
    llm = {"ok": True, "action": "continue", "message": "You seem fully alert!", "next_question": ""}
    assert merge_verdict(rule_verdict(job), llm)["message"] == PASS_MSG


def test_phone_on_another_person_is_not_the_drivers():
    from types import SimpleNamespace
    from fastgears.vision.objects import ObjectDetector
    face = SimpleNamespace(bbox=(250, 100, 330, 200))
    dets = [{"label": "person", "conf": 0.8, "bbox": [150, 80, 450, 480]},   # the driver
            {"label": "person", "conf": 0.6, "bbox": [480, 250, 640, 480]}]  # someone behind, face down
    others = ObjectDetector.other_people(dets, face)
    assert others == [[480, 250, 640, 480]]
    own = [[150, 80, 450, 480]]
    assert ObjectDetector.in_other_person([540, 330, 580, 400], others, own)       # their phone
    assert not ObjectDetector.in_other_person([300, 300, 340, 360], others, own)   # the driver's phone
