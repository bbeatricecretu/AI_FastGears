"""Closed-loop check-in (R2): alert -> question -> measure the reaction -> decide.

After a danger alert ("Eyes on the road!") the co-pilot asks the driver a short question about the
road ahead ("What is directly ahead of you?"). While it waits for the answer, the camera keeps
measuring the driver: how fast the eyes came back to the road, how much of the time they stayed
there, whether the danger is still on. The answer and that reaction are then judged:

  * camera ok       -> the driver is back: end quietly (a short thanks only if they answered)
  * camera not ok   -> say why and ask another question (up to MAX_ROUNDS)
  * still not ok    -> final alert: stop at the next safe place

Two kinds of question, alternated on round 1 (odd check-ins: road, even: check); follow-up rounds
always use a check question:
  * road   "What colour is the car in front?" pulls the eyes back to the road. Nobody can verify the
           answer, so only whether it is a real answer counts.
  * check  "What is seven plus five?" has one right answer, so a wrong one shows the mind isn't back yet.

The camera decides whether to ask again; the answer is optional. A driver who is clearly back on
the road is not asked again just because the mic missed the answer.

The LLM judges the answer (is it a real, coherent answer about the road?), but it can never
pass a driver the camera says is still in danger: the camera rules are a hard floor.

Time-driven (`now` arguments, never the wall clock) and thread-free so it stays unit-testable;
the Coach runs the judge call on its own thread so the video loop never waits for it.
"""
from __future__ import annotations

import json
import re
import threading

ANSWER_TIMEOUT_S = 20.0     # question spoken (~3 s) + time to answer + transcription (~2 s)
MAX_ROUNDS = 3
COOLDOWN_S = 30.0           # after a check-in ends, no new one for this long
REACT_GRACE_S = 2.0         # the first seconds after the question are reaction time, not judged
ON_ROAD_SHARE_OK = 0.6      # eyes on the road at least this share of the judged window
MIN_ANSWER_CHARS = 2

QUESTIONS = (            # road questions: can't be verified, they bring the eyes back
    "What is directly ahead of you right now?",
    "What colour is the vehicle in front of you?",
    "What was the last road sign you passed?",
    "Which lane are you driving in?",
    "Is anything on the road ahead you need to watch for?",
)

# check questions: one right answer, trivial on purpose (a driver must not be overloaded).
# Accepted answers are matched as whole words of the reply, so "it's twelve" passes.
CHECK_QUESTIONS = (
    ("What is seven plus five?", ("12", "twelve")),
    ("What is three plus four?", ("7", "seven")),
    ("What is ten minus two?", ("8", "eight")),
    ("Which day comes after Monday?", ("tuesday",)),
    ("How many wheels does a car have?", ("4", "four")),
    ("What is two plus six?", ("8", "eight")),
)
WRONG_MSG = "That's not right. Stay with me."


def answer_matches(answer: str | None, expected) -> bool:
    words = set(re.findall(r"[a-z0-9]+", (answer or "").lower()))
    return any(e in words for e in expected)


JUDGE_PROMPT = """You are FastGears, an in-car co-pilot running an alertness check, in {language}.
The driver was just in danger ({hazard}). You asked a question about the road. You get ONE combined
picture of the driver: their answer, how fast and how clearly they gave it, and their camera-measured
behaviour while answering. Judge them as a whole: answer AND behaviour together.
- Every measurement comes with a "band" that is already computed: trust the bands, don't re-derive
  them from the numbers. A "fast" or "normal" reply is never too slow.
- "question_type" tells you which kind of question it was:
  * "road": short answers are fine ("a red car", "two lanes", "empty road"). You cannot see the road:
    never doubt whether the answer is true; only whether it is a real answer to the question.
  * "check": it has one right answer, given as "correct_answer". A wrong answer means the driver is
    not yet alert (follow_up). Accept any language or wording ("twelve", "it's 12", "doisprezece").
- Weigh it: a good answer with eyes on the road and no tiredness = alert. A good answer but tired
  behaviour = alert for now, recommend a break. Eyes still off the road, eyes closing, a nonsense
  answer or no answer = not yet alert.
- previous_rounds shows how earlier rounds went: getting better counts in their favour.
Choose the next action:
  "continue"  alert: end the check, short thanks
  "break"     alert enough to continue but tired: end the check, tell them to plan a break soon
  "follow_up" not yet alert: say why in a few words (the co-pilot then asks the next question itself)
  "stop"      clearly unfit (e.g. worse every round): tell them to stop at the next safe place
Reply ONLY with JSON: {{"action": "continue|break|follow_up|stop",
"message": "<at most 15 words to say now; name the real reason; no question>"}}"""

PASS_MSG = "Thanks, you're back with me. Keep your eyes on the road."
BREAK_MSG = "Good answer. Your eyes still show tiredness, so plan a break soon."
BREAK_MSG_NO_ANSWER = "Your eyes still show tiredness. Plan a break soon."
# a check-in after one of these never ends in "continue": answering a question in a 20 s window doesn't
# undo a drowsy episode (the driver sits up to answer), so the best outcome is a break recommendation
DROWSY_HAZARDS = frozenset({"microsleep", "head tilted", "eyes closing", "signs of fatigue"})
STOP_MSG = "You don't seem alert. Please stop at the next safe place and take a break."
SLOW_ANSWER_S, VERY_SLOW_ANSWER_S = 4.0, 8.0   # browser-measured: end of the question -> first word
STAT_CAUTION, STAT_DANGER = 40.0, 70.0          # dashboard score bands (attention.py level_caution / level_danger)


def _on_road(a: dict) -> bool:
    return bool(a.get("face_present", True)) and a.get("looking") == "road" and not a.get("eyes_closed")


def _score_band(v: float) -> str:
    return "high" if v >= STAT_DANGER else "elevated" if v >= STAT_CAUTION else "low"


def evidence(job: dict) -> dict:
    """The one combined picture the LLM judges: answer + reply timing + voice + behaviour statistics.
    Every number carries a band computed here, so the LLM never re-interprets raw numbers. The reply
    time is the browser's (question end -> first word), never the server-side latency, which also
    contains the question being read out and the transcription."""
    r, rt, voice = job["reaction"], job.get("reply_time_s"), job.get("voice") or {}
    if rt is None:
        rt_band = "not measured"
    else:
        rt_band = "very slow" if rt >= VERY_SLOW_ANSWER_S else "slow" if rt >= SLOW_ANSWER_S else "fast" if rt < 2 else "normal"
    kind = job.get("kind", "road")
    q = {"question": job["question"], "question_type": kind}
    if kind == "check":
        q["correct_answer"] = (job.get("expected") or ["?"])[0]
    return {
        **q,
        "answer": job.get("answer") or "(no answer)",
        "reply": {"time_s": rt, "band": rt_band,
                  "voice": voice or "typed or not analysed"},
        "behaviour_while_answering": {
            "eyes_on_road_share": r["on_road_share"],
            "eyes_on_road_band": "good" if r["on_road_share"] >= ON_ROAD_SHARE_OK else "poor",
            "attention_level_now": r["level_now"],
            "microsleep": r["microsleep"],
            "eyes_closing": r.get("eyes_closing", False),
            "drowsiness_avg": r.get("drowsiness_avg", 0.0), "drowsiness_band": _score_band(r.get("drowsiness_avg", 0.0)),
            "distraction_avg": r.get("distraction_avg", 0.0), "distraction_band": _score_band(r.get("distraction_avg", 0.0)),
            "tired_over_last_minutes": r.get("tired", False),
        },
        "round": f"{job['round']} of {job['max_rounds']}",
        "previous_rounds": [{"question": p["question"], "answer": p["answer"], "passed": p["ok"]}
                            for p in job.get("previous_rounds", [])],
    }


def rule_verdict(job: dict) -> dict:
    """The same combined picture, judged by fixed rules. Always computed: it is the fallback without
    an LLM, and its hard limits are a floor the LLM can't override."""
    ans = (job.get("answer") or "").strip()
    answered = len(ans) >= MIN_ANSWER_CHARS
    r = job["reaction"]
    # acute behaviour (changes within seconds): is the driver back?
    cv_ok = (r["level_now"] != "danger" and r["on_road_share"] >= ON_ROAD_SHARE_OK and not r["microsleep"]
             and r.get("distraction_avg", 0.0) < STAT_DANGER and not r.get("eyes_closing"))
    if not cv_ok:
        if r["level_now"] == "danger" or r["microsleep"]:
            msg = "You still look unsafe. Eyes on the road."
        elif r.get("eyes_closing"):
            msg = "Your eyes keep closing."
        else:
            msg = "Your eyes keep leaving the road."
        return {"ok": False, "cv_ok": False, "action": "follow_up", "message": msg, "next_question": ""}
    # the reply itself (browser-measured time, voice)
    rt, voice = job.get("reply_time_s"), job.get("voice") or {}
    if answered and rt is not None and rt >= VERY_SLOW_ANSWER_S:
        return {"ok": False, "cv_ok": True, "action": "follow_up",
                "message": f"That took you {rt:.0f} seconds. Stay with me.", "next_question": ""}
    if answered and voice.get("clarity") == "slurred":
        return {"ok": False, "cv_ok": True, "action": "follow_up", "message": "You sound very tired.",
                "next_question": ""}
    # slow statistics (PERCLOS, yawns over 5 min) can't recover in a 20 s check: they don't fail it,
    # they turn the pass into a break recommendation
    if (r.get("tired") or r.get("drowsiness_avg", 0.0) >= STAT_CAUTION
            or job.get("hazard") in DROWSY_HAZARDS):
        v = {"ok": True, "cv_ok": True, "action": "break",
             "message": BREAK_MSG if answered else BREAK_MSG_NO_ANSWER, "next_question": ""}
    else:
        v = {"ok": True, "cv_ok": True, "action": "continue", "message": PASS_MSG if answered else "",
             "next_question": ""}
    # a check question answered wrong: not back yet. Word matching is English-only and literal, so
    # this one is soft: with an LLM present, the LLM judges correctness instead (see merge_verdict)
    if answered and job.get("kind") == "check" and not answer_matches(ans, job.get("expected") or ()):
        return {"ok": False, "cv_ok": True, "action": "follow_up", "message": WRONG_MSG,
                "next_question": "", "wrong_answer": True, "camera_verdict": v}
    return v


def parse_judge(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    action = d.get("action")
    if action not in ("continue", "break", "follow_up", "stop"):
        if not isinstance(d.get("ok"), bool):           # older {"ok": ...} replies
            return None
        action = "continue" if d["ok"] else "follow_up"
    return {"ok": action in ("continue", "break"), "action": action, "message": str(d.get("message") or "")[:160],
            "next_question": str(d.get("next_question") or "")[:160]}


_SEVERITY = {"continue": 0, "break": 1, "follow_up": 2, "stop": 3}


def merge_verdict(rule: dict, llm: dict | None) -> dict:
    """Final decision on the combined picture: the LLM's holistic reading, but never below the rules'
    hard limits (a driver the camera says is not back can't pass; measured tiredness keeps its break)."""
    if llm is None:
        return {**rule, "source": "rules"}
    if rule.get("wrong_answer"):            # the LLM reads the answer better than word matching
        rule = rule["camera_verdict"]
    llm = {**llm, "action": llm.get("action") or ("continue" if llm["ok"] else "follow_up")}
    action = max(rule.get("action", "continue"), llm["action"], key=_SEVERITY.get)
    if rule["ok"] and not llm["ok"]:
        msg = llm["message"] or "That didn't sound like an answer."
    elif not rule["ok"]:
        msg = llm["message"] if not llm["ok"] and llm["message"] else rule["message"]
    elif action == "break" and rule.get("action") == "break" and llm["action"] != "break":
        msg = rule["message"]                # the LLM's thanks would hide the measured tiredness
    elif action == "continue":
        msg = PASS_MSG                       # fixed thanks: the LLM must not claim "you are alert"
    else:
        msg = llm["message"] or rule["message"]
    ok = action in ("continue", "break")
    return {"ok": ok, "cv_ok": rule.get("cv_ok"), "action": action, "message": msg,
            "next_question": "" if ok else llm["next_question"], "source": "llm"}


class CheckIn:
    """State machine: idle -> waiting (question asked) -> judging -> idle | waiting (next round)."""

    def __init__(self, answer_timeout_s: float = ANSWER_TIMEOUT_S, max_rounds: int = MAX_ROUNDS,
                 cooldown_s: float = COOLDOWN_S):
        self.answer_timeout_s, self.max_rounds, self.cooldown_s = answer_timeout_s, max_rounds, cooldown_s
        self.phase = "idle"
        self.id = 0
        self.round = 0
        self.hazard: str | None = None
        self.question: str | None = None
        self.kind = "road"
        self.expected: tuple[str, ...] = ()
        self.asked_at: float | None = None
        self.rounds: list[dict] = []
        self._samples: list[tuple[float, bool, str, bool]] = []   # t, on_road, level, microsleep
        self._stats: list[tuple[float, float, bool, bool]] = []   # drowsiness, distraction, tired, eyes_closing
        self._ended_at = -1e9
        self._q_next = 0
        self._c_next = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    @property
    def waiting(self) -> bool:
        return self.phase == "waiting"

    def can_start(self, now: float, cooldown_s: float | None = None) -> bool:
        """cooldown_s: a shorter cooldown for a danger that is still going on (see Coach.observe)."""
        cd = self.cooldown_s if cooldown_s is None else min(self.cooldown_s, cooldown_s)
        return self.phase == "idle" and now - self._ended_at >= cd

    def _pick_question(self, kind: str) -> tuple[str, str, tuple[str, ...]]:
        if kind == "check":
            q, expected = CHECK_QUESTIONS[self._c_next % len(CHECK_QUESTIONS)]
            self._c_next += 1
            return q, kind, expected
        q = QUESTIONS[self._q_next % len(QUESTIONS)]
        self._q_next += 1
        return q, kind, ()

    def _ask(self, picked: tuple[str, str, tuple[str, ...]], now: float, lead: str = "") -> dict:
        question, self.kind, self.expected = picked
        self.phase, self.question, self.asked_at = "waiting", question, now
        self._samples, self._stats = [], []
        return {"message": f"{lead} {question}".strip(), "severity": "warning", "reason": "check-in",
                "source": "check", "expects_answer": True, "check_id": self.id, "round": self.round}

    def start(self, hazard: str, now: float, cooldown_s: float | None = None) -> dict | None:
        with self._lock:
            if not self.can_start(now, cooldown_s):
                return None
            self.id += 1
            self.round, self.hazard, self.rounds = 1, hazard, []
            # round 1 alternates: odd check-ins a road question, even ones a check question
            return self._ask(self._pick_question("road" if self.id % 2 else "check"), now)

    def observe(self, state: dict, now: float, hold: bool = False) -> dict | None:
        """Record the driver's reaction; returns a judge job when the answer time runs out.
        hold: the driver is answering right now (speaking / being transcribed): don't time out."""
        with self._lock:
            if self.phase != "waiting":
                return None
            a = state.get("attention", {})
            self._samples.append((now, _on_road(a), a.get("level", "ok"), bool(a.get("microsleep"))))
            self._stats.append((float(a.get("drowsiness") or 0.0), float(a.get("distraction") or 0.0),
                                bool(a.get("tired")), bool(a.get("eyes_closing"))))
            if not hold and now - self.asked_at >= self.answer_timeout_s:
                return self._close(None, now)
            return None

    def answer(self, text: str, now: float) -> dict | None:
        with self._lock:
            if self.phase != "waiting":
                return None
            return self._close(text, now)

    def _reaction(self, now: float) -> dict:
        s, st = self._samples, self._stats
        judged = [x for x in s if x[0] - self.asked_at >= REACT_GRACE_S] or s
        share = sum(x[1] for x in judged) / len(judged) if judged else 0.0
        back = next((x[0] - self.asked_at for x in s if x[1]), None)
        return {"on_road_share": round(share, 2),
                "back_on_road_s": None if back is None else round(back, 1),
                "level_now": s[-1][2] if s else "ok",
                "microsleep": any(x[3] for x in s),
                "watched_s": round(now - self.asked_at, 1),
                # the driver's behaviour statistics over the answer window (same scores as the dashboard)
                "drowsiness_avg": round(sum(x[0] for x in st) / len(st), 1) if st else 0.0,
                "distraction_avg": round(sum(x[1] for x in st) / len(st), 1) if st else 0.0,
                "tired": any(x[2] for x in st),
                "eyes_closing": any(x[3] for x in st)}

    def _close(self, answer: str | None, now: float) -> dict:
        self.phase = "judging"
        return {"check_id": self.id, "round": self.round, "max_rounds": self.max_rounds,
                "hazard": self.hazard, "question": self.question, "answer": answer,
                "kind": self.kind, "expected": list(self.expected),
                "answer_latency_s": None if answer is None else round(now - self.asked_at, 1),
                "reaction": self._reaction(now), "previous_rounds": list(self.rounds)}

    def resolve(self, job: dict, verdict: dict, now: float) -> list[dict]:
        """Apply a verdict to the round it was made for; returns the messages to say."""
        with self._lock:
            if self.phase != "judging" or job["check_id"] != self.id or job["round"] != self.round:
                return []
            self.rounds.append({"question": job["question"], "answer": job["answer"], "ok": verdict["ok"]})
            src = verdict.get("source", "rules")
            if verdict["ok"]:
                self._end(now)
                if not verdict["message"]:                 # passed on the camera alone: stay quiet
                    return []
                brk = verdict.get("action") == "break"     # alert, but the behaviour stats say tired
                return [{"message": verdict["message"], "severity": "warning" if brk else "info",
                         "reason": "take a break" if brk else "check-in passed", "source": src}]
            if verdict.get("action") == "stop" or self.round >= self.max_rounds:
                self._end(now)
                return [{"message": STOP_MSG, "severity": "alert", "reason": "check-in failed", "source": src}]
            self.round += 1
            # follow-ups are always check questions: their answer can be verified
            return [{**self._ask(self._pick_question("check"), now, lead=verdict["message"]), "source": src}]

    def _end(self, now: float) -> None:
        self.phase, self.question, self.asked_at = "idle", None, None
        self.kind, self.expected = "road", ()
        self._ended_at = now

    def status(self, now: float) -> dict:
        with self._lock:
            if self.phase == "idle":
                return {"active": False}
            left = None
            if self.phase == "waiting":
                left = max(0.0, round(self.answer_timeout_s - (now - self.asked_at), 1))
            return {"active": True, "phase": self.phase, "round": self.round, "max_rounds": self.max_rounds,
                    "hazard": self.hazard, "question": self.question, "question_type": self.kind,
                    "seconds_left": left}
