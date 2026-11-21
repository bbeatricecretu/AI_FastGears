"""Driver replies (R2): hear the driver's spoken answer, judge how alert it sounds, act on it.

The co-pilot asks questions to keep a driver engaged; the *way* the driver answers is a signal too:

  * reply time      seconds from the end of the question to the first word (measured in the browser)
  * voice           Gemini hears the audio itself: clear / mumbled / slurred, energy, hesitation
  * content         does the answer make sense for the question (LLM), or is it one word / nothing
  * camera          the attention level while answering (eyes off the road while talking)

`reply_signals` turns the measurable parts into rules (works offline). The LLM adds its own reading
of the content and voice; `merge_assessment` keeps the worse of the two, so the LLM can make the
verdict stricter but never excuse a very slow or slurred reply. `Engagement` keeps the recent
verdicts: two poor replies in a row make the co-pilot suggest a break even if each one alone
looked borderline.

Time-free and thread-free so it stays unit-testable.
"""
from __future__ import annotations

import json
from collections import deque

SLOW_REPLY_S = 4.0       # a reply that starts later than this is slow
VERY_SLOW_REPLY_S = 8.0  # ...and this late is poor
POOR_STREAK = 2          # this many poor replies in a row -> suggest a break

RANK = {"alert": 0, "slow": 1, "poor": 2}
SCORE = {"alert": 90, "slow": 60, "poor": 25}
ACTIONS = ("continue", "follow_up", "break", "stop")

TRANSCRIBE_PROMPT = """Transcribe what the driver says in this short in-car recording ({language}).
Then describe HOW it sounds, for a driver-alertness check (not a medical judgement).
Reply ONLY with JSON:
{{"transcript": "<exact words, empty if no speech>",
  "clarity": "clear|mumbled|slurred",
  "energy": "normal|low",
  "hesitant": true|false}}"""

ANALYSIS_RULES = """
Also judge how alert the driver's LAST reply shows them to be. You get: the reply, how long they
took to start answering (reply_time_s), how their voice sounded (voice) and the camera state.
- "alert": a coherent, relevant answer, given promptly.
- "slow": late, very short, vague or hesitant, but sensible.
- "poor": no real answer, off-topic, confused, slurred, or very late.
Then pick an action:
- "continue": keep chatting normally.
- "follow_up": the reply was weak: ask one more short, easy question that needs a real answer
  (counting, the road ahead, the trip) to wake them up.
- "break": tell them to plan a break at the next rest area.
- "stop": they sound unfit to drive: tell them to stop at the next safe place.
Reply ONLY with JSON: {"message": "<what you say now>", "alertness": "alert|slow|poor",
"signals": ["<2-4 word reasons>"], "action": "continue|follow_up|break|stop"}"""


def parse_voice(text: str) -> dict:
    """Transcription JSON -> {"transcript", "voice": {...}}. Plain text is taken as the transcript."""
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[s:e + 1]) if s >= 0 and e > s else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        return {"transcript": text.strip('"')[:500], "voice": {}}
    voice = {}
    if d.get("clarity") in ("clear", "mumbled", "slurred"):
        voice["clarity"] = d["clarity"]
    if d.get("energy") in ("normal", "low"):
        voice["energy"] = d["energy"]
    if isinstance(d.get("hesitant"), bool):
        voice["hesitant"] = d["hesitant"]
    return {"transcript": str(d.get("transcript") or "").strip()[:500], "voice": voice}


def reply_signals(text: str, reply_time_s: float | None, voice: dict | None, state: dict | None) -> dict:
    """Rule-based reading of one reply: alertness + the reasons for it."""
    voice = voice or {}
    a = (state or {}).get("attention", {})
    words = len((text or "").split())
    signals, worst = [], "alert"

    def mark(level: str, why: str):
        nonlocal worst
        signals.append(why)
        if RANK[level] > RANK[worst]:
            worst = level

    if reply_time_s is not None:
        if reply_time_s >= VERY_SLOW_REPLY_S:
            mark("poor", f"very slow reply {reply_time_s:.1f} s")
        elif reply_time_s >= SLOW_REPLY_S:
            mark("slow", f"slow reply {reply_time_s:.1f} s")
    if words == 0:
        mark("poor", "no answer")
    elif words == 1:
        mark("slow", "one-word answer")
    if voice.get("clarity") == "slurred":
        mark("poor", "slurred speech")
    elif voice.get("clarity") == "mumbled":
        mark("slow", "mumbled speech")
    if voice.get("energy") == "low":
        mark("slow", "low voice energy")
    if voice.get("hesitant"):
        mark("slow", "hesitant")
    if a.get("level") == "danger":
        mark("poor", "danger while answering")
    elif a.get("distracted") or a.get("eyes_closing"):
        mark("slow", "eyes off road while answering")
    if a.get("tired"):      # slow statistic (PERCLOS / yawns): noted with the reply, judged together by the LLM
        signals.append("tired eyes over the last minutes")
    return {"alertness": worst, "signals": signals}


def parse_assessment(text: str) -> dict | None:
    """LLM reply JSON -> {"message", "alertness", "signals", "action"}; plain text -> message only."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return {"message": text[:400]} if text else None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return {"message": text[:400]}
    if not isinstance(d, dict) or not d.get("message"):
        return None
    out = {"message": str(d["message"])[:400]}
    if d.get("alertness") in RANK:
        out["alertness"] = d["alertness"]
    if isinstance(d.get("signals"), list):
        out["signals"] = [str(x)[:40] for x in d["signals"][:4]]
    if d.get("action") in ACTIONS:
        out["action"] = d["action"]
    return out


def merge_assessment(rule: dict, llm: dict | None) -> dict:
    """Worse of the rules and the LLM; the action follows the final alertness."""
    llm = llm or {}
    level = max(rule["alertness"], llm.get("alertness", "alert"), key=RANK.get)
    signals = list(dict.fromkeys(rule["signals"] + llm.get("signals", [])))[:5]
    default = {"alert": "continue", "slow": "follow_up", "poor": "follow_up"}[level]
    action = llm.get("action", default)
    if level != "alert" and action == "continue":   # a weak reply always gets a follow-up at least
        action = "follow_up"
    return {"alertness": level, "score": SCORE[level], "signals": signals, "action": action}


class Engagement:
    """Rolling record of how alert the driver's replies were."""

    def __init__(self, keep: int = 10):
        self.recent: deque = deque(maxlen=keep)

    def add(self, assessment: dict, reply_time_s: float | None) -> dict:
        """Record one reply; escalates the action when replies stay poor. Returns the final assessment."""
        self.recent.append({"alertness": assessment["alertness"], "reply_time_s": reply_time_s})
        streak = self.poor_streak()
        if streak >= POOR_STREAK and assessment["action"] in ("continue", "follow_up"):
            assessment = {**assessment, "action": "break",
                          "signals": assessment["signals"] + [f"{streak} poor replies in a row"]}
        return assessment

    def poor_streak(self) -> int:
        n = 0
        for r in reversed(self.recent):
            if r["alertness"] != "poor":
                break
            n += 1
        return n

    def status(self) -> dict:
        if not self.recent:
            return {"replies": 0}
        last = self.recent[-1]
        times = [r["reply_time_s"] for r in self.recent if r["reply_time_s"] is not None]
        return {"replies": len(self.recent), "last": last["alertness"],
                "score": round(sum(SCORE[r["alertness"]] for r in self.recent) / len(self.recent)),
                "avg_reply_s": round(sum(times) / len(times), 1) if times else None,
                "last_reply_s": last["reply_time_s"], "poor_streak": self.poor_streak()}
