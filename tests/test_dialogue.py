import json
from fastgears.llm.coach import Coach
from fastgears.llm.dialogue import (Engagement, merge_assessment, parse_assessment, parse_voice,
                                    reply_signals)


def st(**a):
    return {"attention": {"calibrated": True, "level": "ok", "reasons": [], **a}, "uptime_s": 60}


def test_parse_voice_json_and_plain_text():
    v = parse_voice('{"transcript": " a red car ", "clarity": "slurred", "energy": "low", "hesitant": true}')
    assert v == {"transcript": "a red car", "voice": {"clarity": "slurred", "energy": "low", "hesitant": True}}
    assert parse_voice('{"transcript": "hi", "clarity": "weird"}')["voice"] == {}
    assert parse_voice("just words")["transcript"] == "just words"


def test_prompt_and_clear_reply_is_alert():
    r = reply_signals("there is a red truck ahead", 1.2, {"clarity": "clear", "energy": "normal"}, st())
    assert r == {"alertness": "alert", "signals": []}


def test_slow_and_poor_replies():
    assert reply_signals("a blue car in front", 5.0, None, st())["alertness"] == "slow"
    poor = reply_signals("uh", 9.5, {"clarity": "slurred"}, st())
    assert poor["alertness"] == "poor" and "slurred speech" in poor["signals"]
    assert reply_signals("", None, None, st())["alertness"] == "poor"
    assert reply_signals("all good here", 1.0, None, st(level="danger"))["alertness"] == "poor"


def test_llm_can_only_make_it_stricter():
    rule = {"alertness": "poor", "signals": ["very slow reply 9.0 s"]}
    m = merge_assessment(rule, {"message": "ok", "alertness": "alert", "action": "continue"})
    assert m["alertness"] == "poor" and m["action"] == "follow_up"           # weak reply: never just "continue"
    m = merge_assessment({"alertness": "alert", "signals": []}, {"message": "x", "alertness": "poor",
                                                                  "signals": ["off-topic"], "action": "break"})
    assert m["alertness"] == "poor" and m["action"] == "break" and m["signals"] == ["off-topic"]


def test_parse_assessment():
    a = parse_assessment('{"message": "Stay with me.", "alertness": "slow", "signals": ["late"], "action": "follow_up"}')
    assert a == {"message": "Stay with me.", "alertness": "slow", "signals": ["late"], "action": "follow_up"}
    assert parse_assessment("Sounds good.") == {"message": "Sounds good."}
    assert parse_assessment('{"message": "hi", "action": "dance"}') == {"message": "hi"}


def test_two_poor_replies_escalate_to_a_break():
    e = Engagement()
    poor = {"alertness": "poor", "score": 25, "signals": [], "action": "follow_up"}
    assert e.add(dict(poor), 9.0)["action"] == "follow_up"
    assert e.add(dict(poor), 10.0)["action"] == "break"
    s = e.status()
    assert s["replies"] == 2 and s["poor_streak"] == 2 and s["avg_reply_s"] == 9.5


class AssessLLM:
    name = "fake"
    def __init__(self, reply): self.reply, self.users = reply, []
    def complete(self, system, user, json_mode=False):
        self.users.append(user)
        return self.reply


def test_chat_sends_voice_and_reply_time_to_the_llm_and_acts_on_it():
    llm = AssessLLM('{"message": "Pull over at the next safe place.", "alertness": "poor", '
                    '"signals": ["confused"], "action": "stop"}')
    c = Coach(llm, lambda m: None)
    out = c.chat("what road", st(), {"clarity": "mumbled"}, 7.5)
    assert '"reply_time_s": 7.5' in llm.users[0] and "mumbled" in llm.users[0]
    assert out["severity"] == "alert" and not out["expects_answer"]
    assert out["message"] == "Pull over at the next safe place."
    assert out["assessment"]["alertness"] == "poor" and out["assessment"]["reply_time_s"] == 7.5


def test_chat_offline_follows_up_on_a_weak_reply():
    c = Coach(None, lambda m: None)
    out = c.chat("uh", st(), {"clarity": "slurred"}, 9.0)
    assert out["expects_answer"] and "Count down" in out["message"]
    out = c.chat("hm", st(), {"clarity": "slurred"}, 9.0)
    assert out["severity"] == "warning" and "break" in out["message"]
    assert c.engagement.status()["poor_streak"] == 2


class _Resp:
    def __init__(self, text): self.text = text


class _Models:
    def __init__(self, fail): self.fail, self.calls = fail, []
    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if model in self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
        return _Resp(f"from {model}")


def test_gemini_falls_back_when_a_model_quota_is_used_up():
    import types as pytypes

    import pytest
    pytest.importorskip("google.genai")
    from fastgears.llm.coach import GeminiClient
    g = GeminiClient("test-key", "lite", ["lite-preview", "flash"])
    g._client = pytypes.SimpleNamespace(models=_Models(fail={"lite"}))
    assert g.complete("sys", "hi") == "from lite-preview" and g.model == "lite-preview"
    assert g.complete("sys", "hi") == "from lite-preview"                 # stays on the working model
    assert g._client.models.calls == ["lite", "lite-preview", "lite-preview"]
    g._client.models.fail = {"lite-preview", "flash"}
    with pytest.raises(RuntimeError, match="429"):                        # all used up: the caller falls back to rules
        g.complete("sys", "hi")


# ── check-in: answer + behaviour judged together ────────────────────────────────

def _job(answer="A red car.", rt=0.6, voice=None, **reaction):
    r = {"on_road_share": 1.0, "back_on_road_s": 0.0, "level_now": "ok", "microsleep": False, "watched_s": 9.4,
         "drowsiness_avg": 5.0, "distraction_avg": 2.0, "tired": False, "eyes_closing": False, **reaction}
    return {"check_id": 1, "round": 1, "max_rounds": 3, "hazard": "eyes off road", "question": "What is ahead?",
            "answer": answer, "answer_latency_s": 9.4, "reaction": r, "previous_rounds": [],
            "voice": voice or {"clarity": "clear"}, "reply_time_s": rt}


def test_evidence_combines_answer_timing_and_behaviour_with_bands():
    from fastgears.llm.intervention import evidence
    e = evidence(_job(rt=0.6, drowsiness_avg=45.0, tired=True))
    assert e["reply"] == {"time_s": 0.6, "band": "fast", "voice": {"clarity": "clear"}}
    b = e["behaviour_while_answering"]
    assert b["drowsiness_band"] == "elevated" and b["tired_over_last_minutes"] and b["eyes_on_road_band"] == "good"
    assert "answer_latency_s" not in json.dumps(e)          # server latency (question + transcription) never shown


def test_rules_fast_answer_passes_tired_behaviour_recommends_a_break():
    from fastgears.llm.intervention import rule_verdict
    assert rule_verdict(_job())["action"] == "continue"
    v = rule_verdict(_job(tired=True))
    assert v["ok"] and v["action"] == "break"
    assert not rule_verdict(_job(on_road_share=0.2))["ok"]               # eyes still off the road
    assert not rule_verdict(_job(rt=9.0))["ok"]                          # measured very slow reply


def test_llm_reading_is_kept_but_never_below_the_measured_floor():
    from fastgears.llm.intervention import PASS_MSG, merge_verdict, parse_judge, rule_verdict
    llm = parse_judge('{"action": "continue", "message": "Thanks, you seem alert."}')
    assert merge_verdict(rule_verdict(_job()), llm)["message"] == PASS_MSG   # never "you seem alert"
    v = merge_verdict(rule_verdict(_job(on_road_share=0.2)), llm)          # camera says not back: no pass
    assert not v["ok"] and v["action"] == "follow_up"
    v = merge_verdict(rule_verdict(_job(tired=True)), llm)                 # measured tiredness keeps its break
    assert v["ok"] and v["action"] == "break" and "break" in v["message"]
    v = merge_verdict(rule_verdict(_job()), parse_judge('{"action": "stop", "message": "Pull over."}'))
    assert not v["ok"] and v["action"] == "stop"


def test_tired_pass_is_a_warning_and_stop_ends_the_check():
    from fastgears.llm.intervention import CheckIn
    ck = CheckIn(cooldown_s=0)
    ck.start("eyes off road", 0.0)
    job = ck.answer("a red car", 2.0)
    out = ck.resolve(job, {"ok": True, "action": "break", "message": "Plan a break soon."}, 3.0)
    assert out[0]["severity"] == "warning" and out[0]["reason"] == "take a break" and not ck.active
    ck.start("eyes off road", 10.0)
    job = ck.answer("banana", 12.0)
    out = ck.resolve(job, {"ok": False, "action": "stop", "message": "x", "next_question": ""}, 13.0)
    assert out[0]["reason"] == "check-in failed" and not ck.active
