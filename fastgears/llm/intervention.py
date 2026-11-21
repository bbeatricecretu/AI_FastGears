"""Closed-loop check-in (R2): alert -> question -> measure the reaction -> decide.

After a danger alert ("Eyes on the road!") the co-pilot asks the driver a short question about the
road ahead ("What is directly ahead of you?"). While it waits for the answer, the camera keeps
measuring the driver: how fast the eyes came back to the road, how much of the time they stayed
there, whether the danger is still on. The answer and that reaction are then judged:

  * camera ok       -> the driver is back: end quietly (a short thanks only if they answered)
  * camera not ok   -> say why and ask another question (up to MAX_ROUNDS)
  * still not ok    -> final alert: stop at the next safe place

The camera decides whether to ask again; the answer is optional. A driver who is clearly back on
the road is not asked again just because the mic missed the answer.

The LLM judges the answer (is it a real, coherent answer about the road?), but it can never
pass a driver the camera says is still in danger: the camera rules are a hard floor.

Time-driven (`now` arguments, never the wall clock) and thread-free so it stays unit-testable;
the Coach runs the judge call on its own thread so the video loop never waits for it.
"""
from __future__ import annotations

import json
import threading

ANSWER_TIMEOUT_S = 12.0     # question spoken (~3 s) + time to answer
MAX_ROUNDS = 3
COOLDOWN_S = 30.0           # after a check-in ends, no new one for this long
REACT_GRACE_S = 2.0         # the first seconds after the question are reaction time, not judged
ON_ROAD_SHARE_OK = 0.6      # eyes on the road at least this share of the judged window
MIN_ANSWER_CHARS = 2

QUESTIONS = (
    "What is directly ahead of you right now?",
    "What colour is the vehicle in front of you?",
    "What was the last road sign you passed?",
    "Which lane are you driving in?",
    "Is anything on the road ahead you need to watch for?",
)

JUDGE_PROMPT = """You are FastGears, an in-car co-pilot running an alertness check, in {language}.
The driver was just in danger ({hazard}). You asked a question about the road and measured their
reaction with the camera. Decide if the driver is now alert and attentive.
- A real, coherent answer about the road (even short, e.g. "a red car", "nothing, empty road") is good.
- No answer, nonsense, or an answer that ignores the question is not ok.
- The camera reaction matters: eyes not back on the road, or still in danger, is not ok.
Reply ONLY with JSON: {{"ok": true|false, "message": "<at most 15 words to say to the driver now>",
"next_question": "<if not ok: a new short question about the road ahead, else empty>"}}"""

PASS_MSG = "Thanks, you're back with me. Keep your eyes on the road."
STOP_MSG = "You don't seem alert. Please stop at the next safe place and take a break."


def _on_road(a: dict) -> bool:
    return bool(a.get("face_present", True)) and a.get("looking") == "road" and not a.get("eyes_closed")


def rule_verdict(job: dict) -> dict:
    """Camera + answer rules. Always computed; the LLM can only make the verdict stricter."""
    ans = (job.get("answer") or "").strip()
    answered = len(ans) >= MIN_ANSWER_CHARS
    r = job["reaction"]
    cv_ok = r["level_now"] != "danger" and r["on_road_share"] >= ON_ROAD_SHARE_OK and not r["microsleep"]
    if cv_ok:   # no answer but eyes back on the road: pass without a word
        return {"ok": True, "cv_ok": True, "message": PASS_MSG if answered else "", "next_question": ""}
    if r["level_now"] == "danger" or r["microsleep"]:
        msg = "You still look unsafe. Eyes on the road."
    else:
        msg = "Your eyes keep leaving the road."
    return {"ok": False, "cv_ok": cv_ok, "message": msg, "next_question": ""}


def parse_judge(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d.get("ok"), bool):
        return None
    return {"ok": d["ok"], "message": str(d.get("message") or "")[:160],
            "next_question": str(d.get("next_question") or "")[:160]}


def merge_verdict(rule: dict, llm: dict | None) -> dict:
    """Final verdict: ok only if the camera rules AND the LLM (when available) agree."""
    if llm is None:
        return {**rule, "source": "rules"}
    ok = rule["ok"] and llm["ok"]
    if ok:
        msg = llm["message"] or rule["message"]
    elif not rule["ok"]:
        msg = rule["message"] if llm["ok"] or not llm["message"] else llm["message"]
    else:                                   # camera fine, but the LLM rejected the answer
        msg = llm["message"] or "That didn't sound like an answer."
    return {"ok": ok, "cv_ok": rule.get("cv_ok"), "message": msg,
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
        self.asked_at: float | None = None
        self.rounds: list[dict] = []
        self._samples: list[tuple[float, bool, str, bool]] = []   # t, on_road, level, microsleep
        self._ended_at = -1e9
        self._q_next = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.phase != "idle"

    @property
    def waiting(self) -> bool:
        return self.phase == "waiting"

    def can_start(self, now: float) -> bool:
        return self.phase == "idle" and now - self._ended_at >= self.cooldown_s

    def _pick_question(self) -> str:
        q = QUESTIONS[self._q_next % len(QUESTIONS)]
        self._q_next += 1
        return q

    def _ask(self, question: str, now: float, lead: str = "") -> dict:
        self.phase, self.question, self.asked_at = "waiting", question, now
        self._samples = []
        return {"message": f"{lead} {question}".strip(), "severity": "warning", "reason": "check-in",
                "source": "check", "expects_answer": True, "check_id": self.id, "round": self.round}

    def start(self, hazard: str, now: float) -> dict | None:
        with self._lock:
            if not self.can_start(now):
                return None
            self.id += 1
            self.round, self.hazard, self.rounds = 1, hazard, []
            return self._ask(self._pick_question(), now)

    def observe(self, state: dict, now: float) -> dict | None:
        """Record the driver's reaction; returns a judge job when the answer time runs out."""
        with self._lock:
            if self.phase != "waiting":
                return None
            a = state.get("attention", {})
            self._samples.append((now, _on_road(a), a.get("level", "ok"), bool(a.get("microsleep"))))
            if now - self.asked_at >= self.answer_timeout_s:
                return self._close(None, now)
            return None

    def answer(self, text: str, now: float) -> dict | None:
        with self._lock:
            if self.phase != "waiting":
                return None
            return self._close(text, now)

    def _reaction(self, now: float) -> dict:
        s = self._samples
        judged = [x for x in s if x[0] - self.asked_at >= REACT_GRACE_S] or s
        share = sum(x[1] for x in judged) / len(judged) if judged else 0.0
        back = next((x[0] - self.asked_at for x in s if x[1]), None)
        return {"on_road_share": round(share, 2),
                "back_on_road_s": None if back is None else round(back, 1),
                "level_now": s[-1][2] if s else "ok",
                "microsleep": any(x[3] for x in s),
                "watched_s": round(now - self.asked_at, 1)}

    def _close(self, answer: str | None, now: float) -> dict:
        self.phase = "judging"
        return {"check_id": self.id, "round": self.round, "max_rounds": self.max_rounds,
                "hazard": self.hazard, "question": self.question, "answer": answer,
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
                return [{"message": verdict["message"], "severity": "info",
                         "reason": "check-in passed", "source": src}]
            if self.round >= self.max_rounds:
                self._end(now)
                return [{"message": STOP_MSG, "severity": "alert", "reason": "check-in failed", "source": src}]
            self.round += 1
            q = verdict.get("next_question") or self._pick_question()
            return [{**self._ask(q, now, lead=verdict["message"]), "source": src}]

    def _end(self, now: float) -> None:
        self.phase, self.question, self.asked_at = "idle", None, None
        self._ended_at = now

    def status(self, now: float) -> dict:
        with self._lock:
            if self.phase == "idle":
                return {"active": False}
            left = None
            if self.phase == "waiting":
                left = max(0.0, round(self.answer_timeout_s - (now - self.asked_at), 1))
            return {"active": True, "phase": self.phase, "round": self.round, "max_rounds": self.max_rounds,
                    "hazard": self.hazard, "question": self.question, "seconds_left": left}
