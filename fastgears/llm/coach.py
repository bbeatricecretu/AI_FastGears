"""R2 — in-car reaction. Turns the CV state into alerts / voice guidance with an LLM.

Design for a live demo:
  * One spoken message per event, never two versions of the same thing:
    - danger ("alert"): the instant local text (zero latency, works offline); the alertness
      question that follows is where the LLM comes in (it judges the answer + behaviour).
    - warning: worded by the LLM for the situation; if it hasn't answered within
      WARN_LLM_WAIT_S the rule text is said instead, and a late LLM reply is dropped.
  * Without a hazard the co-pilot says nothing: no summaries, no praise, no small talk. The only
    "good" message is the thanks after a correctly answered alertness question.
  * One danger episode is one stream of alerts, not one per cause: alerts are at least ALERT_GAP_S
    apart (a microsleep may always interrupt), repeats at least alert_repeat_s apart.
  * Calls run on a worker thread; the video pipeline never waits for the network.
  * Every provider failure falls back to rule-based text, so the demo never goes silent.
Providers: Gemini (google-genai, default), Anthropic (optional), or none.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Callable, Protocol

from .dialogue import (ANALYSIS_RULES, TRANSCRIBE_PROMPT, Engagement, merge_assessment, parse_assessment,
                       parse_voice, reply_signals)
from .intervention import JUDGE_PROMPT, CheckIn, evidence, merge_verdict, parse_judge, rule_verdict

log = logging.getLogger("coach")

ALERT_REPEAT_S = 20.0   # an ongoing danger episode is re-announced at most this often
ALERT_GAP_S = 10.0      # two danger alerts (even for different causes) are at least this far apart
EPISODE_GAP_S = 10.0    # a reason must be gone this long before it counts as a new episode
AWAIT_REPLY_S = 20.0    # after a question the co-pilot stays quiet this long, waiting for the reply
ANSWER_MAX_S = 30.0     # a reply being spoken/transcribed holds the floor at most this long
INTERRUPTS_REPLY = {"microsleep"}   # the only hazard that may talk over an open question
WARN_LLM_WAIT_S = 2.5   # a warning waits this long for the LLM's wording, then the rule text is said
JUDGE_WAIT_S = 6.0      # a check-in verdict waits this long for the LLM, then the camera rules decide
LLM_TIMEOUT_S = 15.0    # network timeout for every LLM call: a stalled connection must not hang a thread

# Message pacing, adjustable live from the dashboard (Coach.set_tuning / POST /api/tuning).
# Danger alerts ignore message_gap_s and warn_hold_s: they are always spoken at once.
#   message_gap_s       a warning waits this long after the previous message (warning or alert)
#   warn_hold_s         a warning must last this long before it is spoken (flicker stays silent)
#   alert_repeat_s      an ongoing danger is re-announced at most this often
#   question_cooldown_s after an alertness-question round ends, no new question for this long
#   questions           ask alertness questions after danger alerts at all
PRESETS = {
    "quiet":     {"message_gap_s": 25.0, "warn_hold_s": 3.0, "alert_repeat_s": 30.0,
                  "question_cooldown_s": 180.0, "questions": True},
    "normal":    {"message_gap_s": 12.0, "warn_hold_s": 1.5, "alert_repeat_s": 20.0,
                  "question_cooldown_s": 90.0, "questions": True},
    "talkative": {"message_gap_s": 5.0, "warn_hold_s": 0.5, "alert_repeat_s": 12.0,
                  "question_cooldown_s": 30.0, "questions": True},
}
TUNING_LIMITS = {"message_gap_s": (0.0, 120.0), "warn_hold_s": (0.0, 10.0), "alert_repeat_s": (5.0, 120.0),
                 "question_cooldown_s": (0.0, 600.0)}

SYSTEM_PROMPT = """You are FastGears, an in-car driver-safety co-pilot. You receive live readings
from a driver-monitoring camera (eyes, head pose, phone use, yawns, camera-measured pulse).
Speak directly to the driver, in {language}. Be short, calm and concrete — the text is read
aloud while driving. Never more than 20 words. Never diagnose medical conditions.
Escalate with the danger: info = friendly, warning = firm, alert = urgent imperative.
Suggest a concrete action when useful (eyes on the road, put the phone away, take a break
at the next rest area, open a window, talk to the passenger). Pulse readings with trust
'unreliable' must be ignored. You only speak because something is wrong: never praise or
reassure ("doing great", "you are alert", "all good").
"primary_hazard" is what matters right now: talk about that one thing only and do not mix in
other readings (e.g. no fatigue advice while the hazard is the phone). Only when it is null may
you comment on slower trends such as yawns or tiredness.
A discomfort_grimace event describes facial movement only. Keep any check-in neutral.
Reply ONLY with JSON: {{"message": "...", "severity": "info|warning|alert", "reason": "<3-6 words>"}}"""

CHAT_PROMPT = """You are FastGears, an in-car co-pilot having a spoken conversation with the driver, in {language}.
Everything you say is read aloud while they drive: at most 25 words, natural and warm, plain text.
You see the live driver-monitoring state and the conversation so far. Keep the thread of the conversation.
Talking keeps a tired driver alert: if the state shows tiredness, end with one short, easy question
(about the trip, music, plans), never one that needs looking at a screen.
Safety first: if the state shows danger, tell them to focus on the road before anything else.
Never praise or reassure them ("doing great", "you seem alert", "driving well"): you can't know that.
If the driver mentions being sleepy, wanting to sleep, being exhausted, drunk or unwell, that comes
first: tell them plainly to stop at the next safe place and rest; don't just carry on chatting.
Treat discomfort_grimace as facial movement only. Never diagnose medical conditions."""

CONVO_TURNS = 8          # conversation lines the LLM sees (driver + co-pilot)
FOLLOW_UP_MSG = "Stay with me. Count down from ten for me, out loud."
BREAK_MSG = "Your answers sound tired. Please plan a break at the next rest area."
STOP_MSG = "You don't sound fit to drive. Please stop at the next safe place."
FATIGUE_OPENER = "You look a bit tired. Talk to me for a moment: where are you heading?"


class LLMClient(Protocol):
    name: str
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


def _quota_or_missing(e: Exception) -> bool:
    """Errors that another model may not have: daily/minute quota used up, model not available."""
    s = str(e)
    return any(k in s for k in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND"))


class GeminiClient:
    """Gemini with model fallback: quotas are per model, so when the main model's free quota is used
    up (429) the next model in the list takes over for the rest of the session."""
    name = "gemini"

    def __init__(self, api_key: str, model: str, fallback_models: tuple[str, ...] | list[str] = ()):
        from google import genai
        from google.genai import types
        self._types = types
        # v1 rejects system_instruction ("Developer instruction is not enabled"): use v1beta
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            api_version="v1beta", timeout=int(LLM_TIMEOUT_S * 1000)))
        self.models = list(dict.fromkeys([model, *fallback_models]))
        self._i = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.models[self._i]

    def _thinking(self, model: str):
        t = self._types
        try:  # models think by default: costs latency we don't want in a car
            return (t.ThinkingConfig(thinking_budget=0) if "2.5" in model
                    else t.ThinkingConfig(thinking_level="minimal"))
        except Exception:
            return None

    def _generate(self, contents, **kw) -> str:
        t = self._types
        i, last = self._i, None
        while i < len(self.models):
            model = self.models[i]
            cfg = dict(kw)
            think = self._thinking(model)
            if think is not None:
                cfg["thinking_config"] = think
            try:
                r = self._client.models.generate_content(model=model, contents=contents,
                                                         config=t.GenerateContentConfig(**cfg))
            except Exception as e:
                if not _quota_or_missing(e):
                    raise
                last = e
                log.warning("Gemini model %s unavailable (%s): trying the next one", model, str(e)[:60])
                i += 1
                continue
            with self._lock:
                if i > self._i:
                    self._i = i
                    log.warning("Gemini: switched to %s for the rest of the session", model)
            return r.text or ""
        raise last or RuntimeError("no Gemini model available")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kw = dict(system_instruction=system, temperature=0.5, max_output_tokens=300)
        if json_mode:
            kw["response_mime_type"] = "application/json"
        return self._generate(user, **kw)

    def transcribe(self, audio: bytes, mime: str, prompt: str) -> str:
        """Speech -> text + how the voice sounds, in one call (Gemini hears the audio itself)."""
        part = self._types.Part.from_bytes(data=audio, mime_type=mime)
        return self._generate([part, prompt], temperature=0.0, max_output_tokens=300,
                              response_mime_type="application/json")


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_S)
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        r = self._client.messages.create(model=self.model, max_tokens=300, system=system,
                                         messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def make_client(settings) -> LLMClient | None:
    p = settings.llm_provider
    try:
        if p == "gemini" and settings.gemini_api_key:
            return GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.gemini_fallback_models)
        if p == "anthropic" and settings.anthropic_api_key:
            return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    except Exception as e:  # missing package, bad key format...
        log.error("LLM provider %s unavailable: %s", p, e)
        return None
    if p != "none":
        log.warning("LLM_PROVIDER=%s but no API key set: using rule-based messages", p)
    return None


# ── rule-based fallback / instant alerts ────────────────────────────────────

def local_message(state: dict) -> dict | None:
    """Immediate message for the current state (no network). None when all is fine.
    Severity follows the attention level: anything in danger is an "alert"."""
    a = state.get("attention", {})
    danger = a.get("level") == "danger"
    sev = "alert" if danger else "warning"
    if a.get("microsleep"):
        return {"message": "Wake up! Eyes on the road now.", "severity": "alert", "reason": "microsleep"}
    if a.get("phone"):
        if danger:
            return {"message": "Put the phone down now and keep both hands on the wheel.", "severity": "alert",
                    "reason": "phone in use"}
        return {"message": "Phone detected. Put it away and keep your eyes on the road.", "severity": "warning",
                "reason": "phone in use"}
    if a.get("looking") in ("tilted", "nodding") and any(
            r.startswith(("head tilted", "head dropping")) for r in a.get("reasons", [])):
        return {"message": "Your head is dropping. Sit up, and take a break if you're tired.",
                "severity": sev, "reason": "head tilted"}
    if a.get("eyes_closing"):
        return {"message": "Open your eyes and stay focused on the road.", "severity": sev,
                "reason": "eyes closing"}
    if a.get("distracted"):
        where = a.get("looking", "away")
        tail = f"Stop looking {where}, look ahead." if where in ("left", "right", "up", "down") else "Look ahead."
        return {"message": f"Too much time with your eyes off the road. {tail}",
                "severity": sev, "reason": "eyes off road"}
    if a.get("hand_off_wheel"):
        return {"message": "Please keep your hands on the wheel.", "severity": sev,
                "reason": "both hands off the wheel"}
    if a.get("tired"):
        return {"message": "You seem tired. Plan a break at the next rest stop.", "severity": sev,
                "reason": "signs of fatigue"}
    if danger:   # a danger with no specific rule above must still be spoken
        return {"message": "Danger! Eyes on the road and both hands on the wheel.", "severity": "alert",
                "reason": "danger"}
    return None


def compact_state(state: dict) -> dict:
    """The part of the state the LLM needs (small prompt = fast + cheap)."""
    a = state.get("attention", {})
    p = state.get("pulse", {}) or {}
    keep = ("level", "reasons", "looking", "off_road_s", "off_road_total_s", "eyes_closed", "eye_closed_s",
            "eyes_closing", "perclos", "tired", "blinks_per_min", "yawns_5min", "microsleep", "phone", "phone_s",
            "drowsiness", "distraction", "face_present")
    out = {k: a.get(k) for k in keep if k in a}
    loc = local_message(state)
    out["primary_hazard"] = loc["reason"] if loc else None   # one topic per message
    if p.get("bpm") and (p.get("trust") or {}).get("label") == "reliable":   # main_final pulse
        out["pulse_bpm"] = p["bpm"]
    event = (state.get("discomfort") or {}).get("event")
    if event:
        out["discomfort_grimace"] = {k: event[k] for k in ("duration_ms", "evidence", "coverage")}
    finding = (state.get("driver_signals") or {}).get("finding")
    if finding and finding not in ("no_combined_pattern", "partial_signals", "face_unavailable"):
        out["biometric_finding"] = finding
    out["driving_minutes"] = round(state.get("uptime_s", 0) / 60, 1)
    return out


def parse_reply(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        return {"message": text[:160], "severity": "info", "reason": "llm"} if text else None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not d.get("message"):
        return None
    if d.get("severity") not in ("info", "warning", "alert"):
        d["severity"] = "info"
    return {"message": str(d["message"])[:200], "severity": d["severity"], "reason": str(d.get("reason", ""))[:60]}


class Coach:
    """Watches the state stream and emits messages via on_message(dict)."""

    def __init__(self, client: LLMClient | None, on_message: Callable[[dict], None],
                 language: str = "English", checkin: CheckIn | None = None,
                 message_gap_s: float = 0.0, warn_hold_s: float = 0.0,
                 alert_repeat_s: float = ALERT_REPEAT_S, await_reply_s: float = AWAIT_REPLY_S):
        self.client, self.on_message = client, on_message
        self.message_gap_s, self.warn_hold_s, self.alert_repeat_s = message_gap_s, warn_hold_s, alert_repeat_s
        self.questions = checkin is not None
        self.preset = "custom"
        self._last_said = -1e9          # last non-danger message (observe clock); message_gap_s counts from it
        self._last_alert = -1e9         # last danger alert (observe clock); ALERT_GAP_S counts from it
        self._episode_start: dict = {}  # reason -> when its current episode began (for warn_hold_s)
        self._llm_episode: dict = {}    # reason -> episode start the LLM already spoke about
        self.language = language
        self.checkin = checkin          # closed-loop alertness check after danger alerts (optional)
        self._t = 0.0                   # time of the last observe(): the clock for answers/verdicts
        self._convo: deque = deque(maxlen=CONVO_TURNS)   # (who, text) lines the LLM sees
        self.history: deque = deque(maxlen=50)
        self.engagement = Engagement()   # how alert the driver's replies sound, over the last few
        self._last_local = {}
        self._last_seen: dict = {}
        self.llm_errors = 0
        self.last_latency_ms: int | None = None
        self.await_reply_s = await_reply_s
        self._await_until = -1e9        # a question is open until then (observe clock)
        self._answering_since: float | None = None

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "rules"

    # -- message pacing ---------------------------------------------------------------------
    def tuning(self) -> dict:
        return {"preset": self.preset, "presets": PRESETS, "limits": TUNING_LIMITS,
                "values": {"message_gap_s": self.message_gap_s, "warn_hold_s": self.warn_hold_s,
                           "alert_repeat_s": self.alert_repeat_s,
                           "question_cooldown_s": self.checkin.cooldown_s if self.checkin else None,
                           "questions": self.questions and self.checkin is not None}}

    def set_tuning(self, preset: str | None = None, **values) -> dict:
        """Apply a preset and/or individual values (numbers clamped to TUNING_LIMITS)."""
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}")
        unknown = set(values) - set(TUNING_LIMITS) - {"questions"}
        if unknown:
            raise ValueError(f"unknown setting(s) {sorted(unknown)}")
        merged = {**(PRESETS[preset] if preset else {}), **{k: v for k, v in values.items() if v is not None}}
        for k, v in merged.items():
            if k == "questions":
                self.questions = bool(v)
                continue
            lo, hi = TUNING_LIMITS[k]
            v = min(hi, max(lo, float(v)))
            if k == "question_cooldown_s":
                if self.checkin:
                    self.checkin.cooldown_s = v
            else:
                setattr(self, k, v)
        now = self.tuning()["values"]
        self.preset = next((n for n, pv in PRESETS.items()
                            if all(now[k] == pv[k] for k in pv if now[k] is not None)), "custom")
        return self.tuning()

    # -- waiting for the driver's reply ------------------------------------------------------
    def answering(self, now: float) -> bool:
        """The driver is speaking / their reply is being transcribed (reported by the browser)."""
        return self._answering_since is not None and now - self._answering_since < ANSWER_MAX_S

    def awaiting_reply(self, now: float) -> bool:
        """A question is open: until the reply arrives (or the window closes) the co-pilot stays quiet."""
        return now < self._await_until or self.answering(now)

    def set_answering(self, active: bool) -> None:
        """Browser: the driver started (True) or abandoned (False) a spoken reply."""
        if active:
            self._answering_since = self._t
        else:
            self._answering_since = None
            # a question still open gets a short grace period, so the driver can try again
            if self._await_until > self._t:
                self._await_until = max(self._await_until, self._t + 3.0)

    def _quiet(self, now: float) -> bool:
        """True while the last message (warning or alert) is younger than message_gap_s."""
        return now - max(self._last_said, self._last_alert) < self.message_gap_s

    def _alert_ready(self, key: str, new: bool, now: float) -> bool:
        """Danger alerts share one pace, whatever their cause: a new cause waits ALERT_GAP_S after the
        last alert, a repeat alert_repeat_s. A microsleep is never held back."""
        if key in INTERRUPTS_REPLY:
            return True
        return now - self._last_alert >= (ALERT_GAP_S if new else self.alert_repeat_s)

    # called by the pipeline at ~5 Hz
    def observe(self, state: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._t = now
        a = state.get("attention", {})
        if not a.get("calibrated", True):
            return
        loc = local_message(state)
        key = loc["reason"] if loc else None

        # 1) instant spoken message, no network: every danger ("alert") and every new warning.
        #    Alerts are re-announced every ALERT_REPEAT_S while they last, a warning once per
        #    episode; an episode ends when its reason has been gone EPISODE_GAP_S. The key holds
        #    the severity too, so a warning that escalates to danger is announced again.
        #    Warnings are paced too: held for warn_hold_s (flicker stays silent) and kept
        #    message_gap_s away from the previous message. All alerts share one pace (_alert_ready),
        #    so one drowsy episode can't fire "eyes closing", "head dropping", "danger" back to back.
        if key:
            if now - self._last_seen.get(key, -1e9) > EPISODE_GAP_S:
                self._episode_start[key] = now
            self._last_seen[key] = now
        for k in [k for k in self._last_local if now - self._last_seen.get(k.split("|")[0], -1e9) > EPISODE_GAP_S]:
            del self._last_local[k]
        alerted = False
        repeated = False    # this alert re-announces a danger that is still going on
        # while a question waits for the driver's reply (or they are answering), the co-pilot keeps
        # measuring but says nothing: only a microsleep, which makes answering impossible, interrupts
        busy = self.awaiting_reply(now) or (self.checkin is not None and self.checkin.active)
        # a danger dropping back to caution is no news: no warning for a hazard already alerted this episode
        downgrade = bool(loc) and loc["severity"] == "warning" and f"{key}|alert" in self._last_local
        if loc and not downgrade and not (busy and key not in INTERRUPTS_REPLY):
            ek = f"{key}|{loc['severity']}"
            is_alert = loc["severity"] == "alert"
            repeat = self.alert_repeat_s if is_alert else float("inf")
            new = ek not in self._last_local
            ready = (self._alert_ready(key, new, now) if is_alert
                     else now - self._episode_start.get(key, now) >= self.warn_hold_s and not self._quiet(now))
            if ready and (ek not in self._last_local or now - self._last_local[ek] > repeat):
                repeated = ek in self._last_local
                self._last_local[ek] = now
                if key == "signs of fatigue":    # tiredness: talking helps more than a warning
                    self._open_conversation(FATIGUE_OPENER, now, loc["severity"], key)
                elif self.client and not is_alert:   # a warning can wait ~1 s for a better wording
                    self._warn_async(loc, compact_state(state), now)
                else:
                    self._emit({**loc, "source": "instant"})
                alerted = is_alert

        # 2) closed loop: after a danger alert ask the driver a question, then judge the reaction
        if self.checkin:
            # a danger still going on when its alert repeats gets a question instead of the same alert
            # again: the long question cooldown then shrinks to the alert repeat time
            cooldown = self.alert_repeat_s if repeated else None
            if alerted and self.questions and self.checkin.can_start(now, cooldown):
                q = self.checkin.start(key, now, cooldown)
                if q:
                    self._emit(q)
            # the answer window doesn't run out while the driver is talking or the reply is being transcribed
            job = self.checkin.observe(state, now, hold=self.answering(now))
            if job:
                self._judge_async(job)
        # nothing else: without a hazard the co-pilot stays silent (no summaries, praise or small talk)

    def _warn_async(self, loc: dict, compact: dict, asked: float) -> None:
        """Say a warning in the LLM's words, or the rule text if the LLM is slow or fails. Exactly one
        of the two is said, and nothing if the situation moved on meanwhile (a question was asked,
        or the hazard escalated to a danger alert)."""
        lock, state = threading.Lock(), {"done": False}
        key = loc["reason"]

        def finish(msg: dict) -> None:
            with lock:
                if state["done"]:
                    return
                state["done"] = True
            if self.awaiting_reply(self._t) or (self.checkin and self.checkin.active):
                return
            if self._last_local.get(f"{key}|alert", -1e9) > asked:   # escalated: the alert said it
                return
            self._emit(msg)

        timer = threading.Timer(WARN_LLM_WAIT_S, lambda: finish({**loc, "source": "instant"}))
        timer.daemon = True
        timer.start()

        def run() -> None:
            out = None
            t0 = time.monotonic()
            try:
                out = parse_reply(self.client.complete(SYSTEM_PROMPT.format(language=self.language),
                                                       "Driver state now:\n" + json.dumps(compact), json_mode=True))
                self.last_latency_ms = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                self.llm_errors += 1
                log.warning("warning wording failed: %s", str(e)[:120])
            if out:
                timer.cancel()
                finish({**out, "severity": loc["severity"], "reason": key, "source": self.provider})
        threading.Thread(target=run, name="coach-warn", daemon=True).start()

    def _open_conversation(self, text: str, now: float, severity: str = "info",
                           reason: str = "conversation") -> None:
        self._convo.append(("co-pilot", text))
        self._emit({"message": text, "severity": severity, "reason": reason, "source": "instant",
                    "expects_answer": True})

    @property
    def can_transcribe(self) -> bool:
        return hasattr(self.client, "transcribe")

    def transcribe(self, audio: bytes, mime: str) -> dict:
        """Driver's recorded reply -> {"transcript", "voice": {clarity, energy, hesitant}}."""
        if not self.can_transcribe:
            raise RuntimeError("speech-to-text needs LLM_PROVIDER=gemini")
        t0 = time.monotonic()
        try:
            out = parse_voice(self.client.transcribe(audio, mime, TRANSCRIBE_PROMPT.format(language=self.language)))
        except Exception:
            self.llm_errors += 1
            raise
        self.last_latency_ms = int((time.monotonic() - t0) * 1000)
        return out

    def chat(self, text: str, state: dict, voice: dict | None = None,
             reply_time_s: float | None = None) -> dict:
        """Synchronous Q&A for the dashboard's 'talk to the co-pilot' box. While a check-in waits
        for an answer, whatever the driver says is that answer. Every reply is also judged for
        alertness (reply time, voice, content, camera) and the co-pilot acts on it."""
        text = text[:500]
        self._answering_since, self._await_until = None, -1e9    # the reply is here: the question is closed
        msg = {"role": "user", "message": text, "severity": "info", "source": "user", "t": time.time()}
        self.history.append(msg)
        rule = reply_signals(text, reply_time_s, voice, state)
        if self.checkin and self.checkin.waiting:
            job = self.checkin.answer(text, self._t)
            if job:
                job["voice"], job["reply_time_s"] = voice or {}, reply_time_s
                job["reply_signals"] = rule["signals"]
                self.engagement.add(merge_assessment(rule, None), reply_time_s)
                self._judge_async(job)
                return {"message": "", "severity": "info", "reason": "check-in answer", "source": "check"}
        talk = "\n".join(f"{who}: {line}" for who, line in self._convo) or "(none)"
        self._convo.append(("driver", text))
        llm = None
        if self.client:
            try:
                about = {"reply_time_s": reply_time_s, "voice": voice or "typed, not spoken"}
                llm = parse_assessment(self.client.complete(
                    CHAT_PROMPT.format(language=self.language) + ANALYSIS_RULES,
                    f"Live state: {json.dumps(compact_state(state))}\n\n"
                    f"Conversation so far:\n{talk}\n\nDriver says: {text}\n"
                    f"How they said it: {json.dumps(about)}", json_mode=True))
            except Exception as e:
                log.warning("chat failed: %s", e)
                self.llm_errors += 1
        assessment = self.engagement.add(merge_assessment(rule, llm), reply_time_s)
        action = assessment["action"]
        if llm and llm.get("action", action) == action:
            reply = llm["message"]
        elif action == "break":
            reply = BREAK_MSG
        elif action == "stop":
            reply = STOP_MSG
        elif llm:
            reply = llm["message"]
        elif action == "follow_up":
            reply = FOLLOW_UP_MSG
        elif self.client:
            reply = "Sorry, I can't reach the assistant right now."
        else:
            reply = "I'm in offline mode (no LLM key). " + (local_message(state) or {"message": "All looks fine — drive safe."})["message"]
        reply = reply.strip()[:400]
        self._convo.append(("co-pilot", reply))
        severity = {"break": "warning", "stop": "alert"}.get(action, "info")
        # expects_answer: the browser reopens the mic, so the driver can just keep talking
        out = {"message": reply, "severity": severity, "reason": {"break": "tired replies", "stop": "unfit replies"}.get(action, "chat"),
               "source": self.provider if llm else "rules", "expects_answer": action in ("continue", "follow_up"),
               "assessment": {**assessment, "reply_time_s": reply_time_s, "voice": voice or {}}}
        self._emit(out)
        return out

    # -- closed-loop check-in ---------------------------------------------------------------
    def judge(self, job: dict) -> dict:
        """Verdict for one check-in round: camera rules, made stricter by the LLM when present."""
        rule = rule_verdict(job)
        llm = None
        if self.client and job.get("answer"):   # nothing to judge without an answer: camera decides
            try:
                txt = self.client.complete(JUDGE_PROMPT.format(language=self.language, hazard=job["hazard"]),
                                           "Check-in round:\n" + json.dumps(evidence(job)), json_mode=True)
                llm = parse_judge(txt)
            except Exception as e:
                self.llm_errors += 1
                log.warning("check-in judge failed: %s", e)
        return merge_verdict(rule, llm)

    def _judge_async(self, job: dict) -> None:
        """Judge one round off the video thread. While a round is being judged the co-pilot stays
        quiet, so the LLM gets at most JUDGE_WAIT_S: then the camera rules decide. Whichever comes
        first is applied; resolve() ignores the second (the round is no longer "judging")."""
        lock, done = threading.Lock(), {"v": False}

        def apply(verdict: dict) -> None:
            with lock:
                if done["v"]:
                    return
                done["v"] = True
            for m in self.checkin.resolve(job, verdict, self._t):
                self._emit(m)

        def fallback() -> None:
            apply(merge_verdict(rule_verdict(job), None))

        timer = threading.Timer(JUDGE_WAIT_S, fallback)
        timer.daemon = True
        timer.start()

        def run():
            try:
                verdict = self.judge(job)
            except Exception:
                log.exception("check-in judge crashed: camera rules decide")
                verdict = merge_verdict(rule_verdict(job), None)
            timer.cancel()
            apply(verdict)
        threading.Thread(target=run, name="checkin-judge", daemon=True).start()

    def checkin_status(self) -> dict:
        return self.checkin.status(self._t) if self.checkin else {"active": False}

    def _emit(self, msg: dict) -> None:
        if msg.get("severity") == "alert":
            self._last_alert = self._t  # alerts pace each other (ALERT_GAP_S) and hold warnings back
        else:
            self._last_said = self._t
        if msg.get("expects_answer"):
            self._await_until = self._t + self.await_reply_s   # quiet until the driver replies
        msg = {**msg, "t": time.time()}
        self.history.append(msg)
        try:
            self.on_message(msg)
        except Exception as e:
            log.warning("on_message failed: %s", e)
