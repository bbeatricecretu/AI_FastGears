"""R2 — in-car reaction. Turns the CV state into alerts / voice guidance with an LLM.

Design for a live demo:
  * Danger events (microsleep, long eyes-off-road, phone) get an *instant* local alert
    (zero latency, works offline), then the LLM's contextual message follows.
  * The LLM is called on state changes, rate-limited (LLM_MIN_INTERVAL_S), plus a calm
    periodic check-in summary (LLM_SUMMARY_EVERY_S) so the dashboard always has fresh text.
  * Calls run on a worker thread; the video pipeline never waits for the network.
  * Every provider failure falls back to rule-based text, so the demo never goes silent.
Providers: Gemini (google-genai, default), Anthropic (optional), or none.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Protocol

from .dialogue import (ANALYSIS_RULES, TRANSCRIBE_PROMPT, Engagement, merge_assessment, parse_assessment,
                       parse_voice, reply_signals)
from .intervention import JUDGE_PROMPT, CheckIn, evidence, merge_verdict, parse_judge, rule_verdict

log = logging.getLogger("coach")

ALERT_REPEAT_S = 20.0   # an ongoing danger episode is re-announced at most this often
EPISODE_GAP_S = 10.0    # a reason must be gone this long before it counts as a new episode
RATE_LIMIT_PAUSE_S = 30.0
AWAIT_REPLY_S = 20.0    # after a question the co-pilot stays quiet this long, waiting for the reply
ANSWER_MAX_S = 30.0     # a reply being spoken/transcribed holds the floor at most this long
INTERRUPTS_REPLY = {"microsleep"}   # the only hazard that may talk over an open question

# Message pacing, adjustable live from the dashboard (Coach.set_tuning / POST /api/tuning).
# Danger alerts ignore message_gap_s and warn_hold_s: they are always spoken at once.
#   message_gap_s       minimum quiet time between two non-danger messages (warnings, LLM, summaries)
#   warn_hold_s         a warning must last this long before it is spoken (flicker stays silent)
#   alert_repeat_s      an ongoing danger is re-announced at most this often
#   summary_every_s     calm periodic check-in text while all is fine
#   question_cooldown_s after an alertness-question round ends, no new question for this long
#   questions           ask alertness questions after danger alerts at all
PRESETS = {
    "quiet":     {"message_gap_s": 25.0, "warn_hold_s": 3.0, "alert_repeat_s": 30.0,
                  "summary_every_s": 180.0, "question_cooldown_s": 180.0, "questions": True},
    "normal":    {"message_gap_s": 12.0, "warn_hold_s": 1.5, "alert_repeat_s": 20.0,
                  "summary_every_s": 90.0, "question_cooldown_s": 90.0, "questions": True},
    "talkative": {"message_gap_s": 5.0, "warn_hold_s": 0.5, "alert_repeat_s": 12.0,
                  "summary_every_s": 40.0, "question_cooldown_s": 30.0, "questions": True},
}
TUNING_LIMITS = {"message_gap_s": (0.0, 120.0), "warn_hold_s": (0.0, 10.0), "alert_repeat_s": (5.0, 120.0),
                 "summary_every_s": (10.0, 600.0), "question_cooldown_s": (0.0, 600.0)}

SYSTEM_PROMPT = """You are FastGears, an in-car driver-safety co-pilot. You receive live readings
from a driver-monitoring camera (eyes, head pose, phone use, yawns, camera-measured pulse).
Speak directly to the driver, in {language}. Be short, calm and concrete — the text is read
aloud while driving. Never more than 20 words. Never diagnose medical conditions.
Escalate with the danger: info = friendly, warning = firm, alert = urgent imperative.
Suggest a concrete action when useful (eyes on the road, put the phone away, take a break
at the next rest area, open a window, talk to the passenger). Pulse readings with trust
'unreliable' must be ignored.
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
Treat discomfort_grimace as facial movement only. Never diagnose medical conditions."""

CONVO_TURNS = 8          # conversation lines the LLM sees (driver + co-pilot)
FOLLOW_UP_MSG = "Stay with me. Count down from ten for me, out loud."
BREAK_MSG = "Your answers sound tired. Please plan a break at the next rest area."
STOP_MSG = "You don't sound fit to drive. Please stop at the next safe place."
FATIGUE_OPENER = "You look a bit tired. Talk to me for a moment: where are you heading?"
OPENERS = ("How is the drive going so far?",
           "How are you feeling? Need a break soon?",
           "Long stretch of road. What are you listening to?")


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
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
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
        self._client = anthropic.Anthropic(api_key=api_key)
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
        where = "away from the road" if where in ("away", "road") else where
        return {"message": f"Too much time with your eyes off the road. Stop looking {where}, look ahead.",
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
                 min_interval_s: float = 8.0, summary_every_s: float = 45.0, language: str = "English",
                 checkin: CheckIn | None = None, convo_idle_s: float = 0.0,
                 message_gap_s: float = 0.0, warn_hold_s: float = 0.0,
                 alert_repeat_s: float = ALERT_REPEAT_S, await_reply_s: float = AWAIT_REPLY_S):
        self.client, self.on_message = client, on_message
        self.min_interval_s, self.summary_every_s = min_interval_s, summary_every_s
        self.message_gap_s, self.warn_hold_s, self.alert_repeat_s = message_gap_s, warn_hold_s, alert_repeat_s
        self.questions = checkin is not None
        self.preset = "custom"
        self._last_said = -1e9          # last non-danger message (observe clock); message_gap_s counts from it
        self._episode_start: dict = {}  # reason -> when its current episode began (for warn_hold_s)
        self._llm_episode: dict = {}    # reason -> episode start the LLM already spoke about
        self.language = language
        self.checkin = checkin          # closed-loop alertness check after danger alerts (optional)
        self._t = 0.0                   # time of the last observe(): the clock for answers/verdicts
        self.convo_idle_s = convo_idle_s  # co-pilot opens a conversation after this long without one (0 = never)
        self._convo: deque = deque(maxlen=CONVO_TURNS)   # (who, text) lines the LLM sees
        self._last_talk: float | None = None
        self._opener_i = 0
        self.history: deque = deque(maxlen=50)
        self.engagement = Engagement()   # how alert the driver's replies sound, over the last few
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._last_llm = 0.0
        self._last_local = {}
        self._last_seen: dict = {}
        self._prev_seen: dict = {}
        self._pending = False
        self._pause_until = -1e9
        self._last_level = "ok"
        self._last_key = None
        self._last_discomfort_id = None
        self.llm_errors = 0
        self.last_latency_ms: int | None = None
        self.await_reply_s = await_reply_s
        self._await_until = -1e9        # a question is open until then (observe clock)
        self._answering_since: float | None = None
        threading.Thread(target=self._worker, name="coach", daemon=True).start()

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "rules"

    # -- message pacing ---------------------------------------------------------------------
    def tuning(self) -> dict:
        return {"preset": self.preset, "presets": PRESETS, "limits": TUNING_LIMITS,
                "values": {"message_gap_s": self.message_gap_s, "warn_hold_s": self.warn_hold_s,
                           "alert_repeat_s": self.alert_repeat_s, "summary_every_s": self.summary_every_s,
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
        """True while the last non-danger message is younger than message_gap_s."""
        return now - self._last_said < self.message_gap_s

    # called by the pipeline at ~5 Hz
    def observe(self, state: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._t = now
        a = state.get("attention", {})
        if not a.get("calibrated", True):
            return
        level = a.get("level", "ok")
        loc = local_message(state)
        key = loc["reason"] if loc else None

        # 1) instant spoken message, no network: every danger ("alert") and every new warning.
        #    Alerts are re-announced every ALERT_REPEAT_S while they last, a warning once per
        #    episode; an episode ends when its reason has been gone EPISODE_GAP_S. The key holds
        #    the severity too, so a warning that escalates to danger is announced again.
        #    Warnings are paced too: held for warn_hold_s (flicker stays silent) and kept
        #    message_gap_s away from the previous non-danger message.
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
            ready = is_alert or (now - self._episode_start.get(key, now) >= self.warn_hold_s
                                 and not self._quiet(now))
            if ready and (ek not in self._last_local or now - self._last_local[ek] > repeat):
                repeated = ek in self._last_local
                self._last_local[ek] = now
                if key == "signs of fatigue":    # tiredness: talking helps more than a warning
                    self._open_conversation(FATIGUE_OPENER, now, loc["severity"], key)
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
        if (self.checkin and self.checkin.active) or self.awaiting_reply(now):
            # the dialogue owns the voice: no LLM summaries, openers or comments meanwhile. Nothing is
            # queued either: after the reply the co-pilot speaks about the state as it is then.
            self._pending = False
            self._last_key, self._last_level = key, level
            if key:
                self._prev_seen[key] = now
            return

        # 3) a quiet, calm drive: the co-pilot starts a short conversation now and then
        if self._last_talk is None:
            self._last_talk = now
        if self.convo_idle_s and level == "ok" and now - self._last_talk >= self.convo_idle_s:
            self._open_conversation(OPENERS[self._opener_i % len(OPENERS)], now)
            self._opener_i += 1

        # 4) LLM on change, or periodic summary
        # ask the LLM on a NEW episode (reason not seen in the last EPISODE_GAP_S) or an escalation,
        # not on every flicker of the same situation
        rank = {"ok": 0, "caution": 1, "danger": 2}
        new_episode = bool(key) and now - self._prev_seen.get(key, -1e9) > EPISODE_GAP_S
        changed = new_episode or rank.get(level, 0) > rank.get(self._last_level, 0)
        if key:
            self._prev_seen[key] = now
        self._pending = (self._pending or changed) and level != "ok"   # waits for min_interval if needed
        due = now - self._last_llm >= (self.min_interval_s if self._pending else self.summary_every_s)
        # the LLM follow-up of a danger alert just spoken may come at once; anything else waits its turn
        due = due and (alerted or not self._quiet(now))
        self._last_key, self._last_level = key, level
        event = (state.get("discomfort") or {}).get("event")
        if (event and event["id"] != self._last_discomfort_id and now - self._last_llm >= self.min_interval_s
                and not self._quiet(now)):
            self._last_discomfort_id = event["id"]
            if loc is None:
                loc = {"message": "Are you comfortable? If needed, take a break when safe.",
                       "severity": "info", "reason": "facial movement"}
            due = True
        # one LLM message per danger episode: the instant alert repeats are enough after that, and
        # caution/danger flicker within the episode must not make the LLM say the same thing again
        if key and not event and self._llm_episode.get(key) == self._episode_start.get(key):
            due = False
        if due and now >= self._pause_until:
            self._last_llm = now
            self._pending = False
            if key:
                self._llm_episode[key] = self._episode_start.get(key)
            self._submit(("react", compact_state(state), loc))

    def _submit(self, job) -> None:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:          # one in flight is enough; drop stale requests
            pass

    def _open_conversation(self, text: str, now: float, severity: str = "info",
                           reason: str = "conversation") -> None:
        self._last_talk = now
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
        self._last_talk = self._t
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
        def run():
            verdict = self.judge(job)
            for m in self.checkin.resolve(job, verdict, self._t):
                self._emit(m)
        threading.Thread(target=run, name="checkin-judge", daemon=True).start()

    def checkin_status(self) -> dict:
        return self.checkin.status(self._t) if self.checkin else {"active": False}

    def _worker(self) -> None:
        while True:
            _, compact, loc = self._jobs.get()
            out = None
            if self.client:
                t0 = time.monotonic()
                try:
                    txt = self.client.complete(SYSTEM_PROMPT.format(language=self.language),
                                               "Driver state now:\n" + json.dumps(compact), json_mode=True)
                    out = parse_reply(txt)
                    self.last_latency_ms = int((time.monotonic() - t0) * 1000)
                except Exception as e:
                    self.llm_errors += 1
                    log.warning("LLM call failed: %s", e)
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        self._pause_until = time.monotonic() + RATE_LIMIT_PAUSE_S
                        continue            # rate-limited: stay quiet, instant alerts still work
            if self.awaiting_reply(self._t) or (self.checkin and self.checkin.active):
                continue                    # a question was asked meanwhile: this comment would talk over it
            if loc and out is not None:     # the LLM spoke about this hazard: it counts as its announcement,
                for sev in ("warning", "alert"):   # so the instant repeat waits (existing keys only:
                    ek = f"{loc['reason']}|{sev}"  # observe() iterates this dict on its own thread)
                    if ek in self._last_local:
                        self._last_local[ek] = self._t
            if out is None:
                out = loc or {"message": "All good. Keep your eyes on the road.", "severity": "info", "reason": "ok"}
                out = {**out, "source": "rules"}
            else:
                out["source"] = self.provider
            self._emit(out)

    def _emit(self, msg: dict) -> None:
        if not (msg.get("severity") == "alert" and msg.get("source") == "instant"):
            self._last_said = self._t   # a danger alert doesn't make the next warning wait
        if msg.get("expects_answer"):
            self._await_until = self._t + self.await_reply_s   # quiet until the driver replies
        msg = {**msg, "t": time.time()}
        self.history.append(msg)
        try:
            self.on_message(msg)
        except Exception as e:
            log.warning("on_message failed: %s", e)
