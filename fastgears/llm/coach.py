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

log = logging.getLogger("coach")

ALERT_REPEAT_S = 20.0   # an ongoing danger episode is re-announced at most this often
EPISODE_GAP_S = 10.0    # a reason must be gone this long before it counts as a new episode
RATE_LIMIT_PAUSE_S = 30.0

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
Reply ONLY with JSON: {{"message": "...", "severity": "info|warning|alert", "reason": "<3-6 words>"}}"""

CHAT_PROMPT = """You are FastGears, an in-car co-pilot talking with the driver or a passenger, in {language}.
You see the live driver-monitoring state below. Answer in at most 2 short sentences, safety first.
If the driver seems drowsy or distracted, gently say so. Plain text, no markdown."""


class LLMClient(Protocol):
    name: str
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._types = types
        # v1 rejects system_instruction ("Developer instruction is not enabled"): use v1beta
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        t = self._types
        kw = dict(system_instruction=system, temperature=0.5, max_output_tokens=300)
        if json_mode:
            kw["response_mime_type"] = "application/json"
        try:  # models think by default: costs latency we don't want in a car
            kw["thinking_config"] = (t.ThinkingConfig(thinking_budget=0) if "2.5" in self.model
                                     else t.ThinkingConfig(thinking_level="minimal"))
        except Exception:
            pass
        r = self._client.models.generate_content(model=self.model, contents=user,
                                                 config=t.GenerateContentConfig(**kw))
        return r.text or ""


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
            return GeminiClient(settings.gemini_api_key, settings.gemini_model)
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
    """Immediate message for the current state (no network). None when all is fine."""
    a = state.get("attention", {})
    if a.get("microsleep"):
        return {"message": "Wake up! Eyes on the road now.", "severity": "alert", "reason": "microsleep"}
    if a.get("phone"):
        return {"message": "Put the phone down and keep both hands on the wheel.", "severity": "alert",
                "reason": "phone in use"}
    if a.get("looking") in ("tilted", "nodding") and any(
            r.startswith(("head tilted", "head dropping")) for r in a.get("reasons", [])):
        return {"message": "Your head is dropping. Sit up, and take a break if you're tired.",
                "severity": "alert" if a.get("level") == "danger" else "warning", "reason": "head tilted"}
    if a.get("distracted"):
        where = a.get("looking", "away")
        return {"message": f"You're looking {where}. Eyes back on the road.", "severity": "warning",
                "reason": f"looking {where}"}
    if a.get("hand_off_wheel"):
        return {"message": "Please keep both hands on the wheel.", "severity": "warning",
                "reason": "hand off the wheel"}
    if a.get("yawning") or (a.get("perclos") or 0) > 0.15:
        return {"message": "You seem tired. Plan a break at the next rest stop.", "severity": "warning",
                "reason": "signs of fatigue"}
    return None


def compact_state(state: dict) -> dict:
    """The part of the state the LLM needs (small prompt = fast + cheap)."""
    a = state.get("attention", {})
    p = state.get("pulse", {}) or {}
    keep = ("level", "reasons", "looking", "off_road_s", "eyes_closed", "eye_closed_s", "perclos",
            "blinks_per_min", "yawns_5min", "microsleep", "phone", "drowsiness", "distraction", "face_present")
    out = {k: a.get(k) for k in keep if k in a}
    loc = local_message(state)
    out["primary_hazard"] = loc["reason"] if loc else None   # one topic per message
    if p.get("bpm") and p.get("pulse_state") in ("tracking", "held"):
        out["pulse_bpm"] = p["bpm"]
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
                 min_interval_s: float = 8.0, summary_every_s: float = 45.0, language: str = "English"):
        self.client, self.on_message = client, on_message
        self.min_interval_s, self.summary_every_s = min_interval_s, summary_every_s
        self.language = language
        self.history: deque = deque(maxlen=50)
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        self._last_llm = 0.0
        self._last_local = {}
        self._last_seen: dict = {}
        self._prev_seen: dict = {}
        self._pending = False
        self._pause_until = -1e9
        self._last_level = "ok"
        self._last_key = None
        self.llm_errors = 0
        self.last_latency_ms: int | None = None
        threading.Thread(target=self._worker, name="coach", daemon=True).start()

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "rules"

    # called by the pipeline at ~5 Hz
    def observe(self, state: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        a = state.get("attention", {})
        if not a.get("calibrated", True):
            return
        level = a.get("level", "ok")
        loc = local_message(state)
        key = loc["reason"] if loc else None

        # 1) instant local alert once per danger episode (re-announced every ALERT_REPEAT_S
        #    while it lasts); an episode ends when its reason disappears
        if key:
            self._last_seen[key] = now
        for k in [k for k in self._last_local if now - self._last_seen.get(k, -1e9) > EPISODE_GAP_S]:
            del self._last_local[k]
        if loc and loc["severity"] == "alert":
            if now - self._last_local.get(key, -1e9) > ALERT_REPEAT_S:
                self._last_local[key] = now
                self._emit({**loc, "source": "instant"})

        # 2) LLM on change, or periodic summary
        # ask the LLM on a NEW episode (reason not seen in the last EPISODE_GAP_S) or an escalation,
        # not on every flicker of the same situation
        rank = {"ok": 0, "caution": 1, "danger": 2}
        new_episode = bool(key) and now - self._prev_seen.get(key, -1e9) > EPISODE_GAP_S
        changed = new_episode or rank.get(level, 0) > rank.get(self._last_level, 0)
        if key:
            self._prev_seen[key] = now
        self._pending = (self._pending or changed) and level != "ok"   # waits for min_interval if needed
        due = now - self._last_llm >= (self.min_interval_s if self._pending else self.summary_every_s)
        self._last_key, self._last_level = key, level
        if due and now >= self._pause_until:
            self._last_llm = now
            self._pending = False
            self._submit(("react", compact_state(state), loc))

    def _submit(self, job) -> None:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:          # one in flight is enough; drop stale requests
            pass

    def chat(self, text: str, state: dict) -> dict:
        """Synchronous Q&A for the dashboard's 'talk to the co-pilot' box."""
        msg = {"role": "user", "message": text[:500], "severity": "info", "source": "user", "t": time.time()}
        self.history.append(msg)
        if not self.client:
            reply = "I'm in offline mode (no LLM key). " + (local_message(state) or {"message": "All looks fine — drive safe."})["message"]
        else:
            try:
                reply = self.client.complete(CHAT_PROMPT.format(language=self.language),
                                             f"Live state: {json.dumps(compact_state(state))}\n\nDriver/passenger says: {text}")
            except Exception as e:
                log.warning("chat failed: %s", e)
                self.llm_errors += 1
                reply = "Sorry, I can't reach the assistant right now."
        out = {"message": reply.strip()[:400], "severity": "info", "reason": "chat", "source": self.provider}
        self._emit(out)
        return out

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
            if out is None:
                out = loc or {"message": "All good. Keep your eyes on the road.", "severity": "info", "reason": "ok"}
                out = {**out, "source": "rules"}
            else:
                out["source"] = self.provider
            self._emit(out)

    def _emit(self, msg: dict) -> None:
        msg = {**msg, "t": time.time()}
        self.history.append(msg)
        try:
            self.on_message(msg)
        except Exception as e:
            log.warning("on_message failed: %s", e)
