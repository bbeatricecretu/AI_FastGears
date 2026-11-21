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
import re
import threading
import time
from collections import deque
from typing import Callable, Protocol, Literal
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("coach")

SYSTEM_PROMPT = """You are FastGears, an in-car driver-safety co-pilot. You receive live readings
from a driver-monitoring camera (eyes, head pose, phone use, yawns, camera-measured pulse).
Speak directly to the driver, in {language}. Be short, calm and concrete — the text is read
aloud while driving. Never more than 20 words. Never diagnose medical conditions.
Escalate with the danger: info = friendly, warning = firm, alert = urgent imperative.
Suggest a concrete action when useful (eyes on the road, put the phone away, take a break
when safe). Use only the supplied observations. The camera faces the driver. Pulse readings with trust
'unreliable' must be ignored.
A discomfort_grimace event describes facial movement only. Keep any check-in neutral.
Reply ONLY with JSON: {{"message": "...", "severity": "info|warning|alert", "reason": "<3-6 words>"}}"""

CHAT_PROMPT = """You are FastGears, an in-car co-pilot talking with the driver or a passenger, in {language}.
You see the live driver-monitoring state below. Answer in at most 2 short sentences, safety first.
If the driver seems drowsy or distracted, gently say so. Treat discomfort_grimace as a description of
facial movement only. Never diagnose medical conditions. Plain text, no markdown."""


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
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta", timeout=20000,
                                         retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1)))
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False, *, schema=None) -> str:
        t = self._types
        kw = dict(system_instruction=system, temperature=0.5, max_output_tokens=300)
        if json_mode:
            kw["response_mime_type"] = "application/json"
        if schema is not None:
            kw.update(response_json_schema=schema, max_output_tokens=1600)
        try:  # 2.5 models think by default: costs latency we don't want in a car
            kw["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
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
        log.error("LLM provider %s unavailable (%s)", p, type(e).__name__)
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
    posture = state.get("driver_posture", {})
    if posture.get("active"):
        return {"message": "Your head is dropping. Pull over safely if you feel tired.",
                "severity": posture["severity"], "reason": "head dropping"}
    extra_warning = (state.get("driver_attention") or {}).get("active_warning")
    if extra_warning and extra_warning["voice"] and extra_warning["reason"] in ("nod", "severe", "eyes"):
        return {"message": extra_warning["message"], "severity": extra_warning["severity"],
                "reason": "driver drowsiness"}
    if a.get("distracted"):
        where = a.get("looking", "away")
        return {"message": f"You're looking {where}. Eyes back on the road.", "severity": "warning",
                "reason": f"looking {where}"}
    if a.get("level") == "danger":
        return {"message": "Danger: please focus on the road.", "severity": "alert", "reason": "danger"}
    if (state.get("driver_alerts", {}).get("hands") or {}).get("active"):
        return {"message": "Please keep your hands on the wheel.", "severity": "warning",
                "reason": "both hands off the wheel"}
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
    trust = (p.get("trust") or {}).get("label")
    if p.get("bpm") and trust != "unreliable":
        out["pulse_bpm"], out["pulse_trust"] = p["bpm"], trust
    event = (state.get("discomfort") or {}).get("event")
    if event:
        out["discomfort_grimace"] = {k: event[k] for k in ("duration_ms", "evidence", "coverage")}
    alerts = state.get("driver_alerts", {})
    if alerts:
        out["driver_alerts"] = {kind: item["active"] for kind, item in alerts.items()}
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
    if not isinstance(d, dict) or not isinstance(d.get("message"), str) or not d["message"].strip():
        return None
    if d.get("severity") not in ("info", "warning", "alert"):
        d["severity"] = "info"
    return {"message": str(d["message"])[:200], "severity": d["severity"], "reason": str(d.get("reason", ""))[:60]}


class PatternReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern_name: str = Field(max_length=100)
    severity_suggestion: Literal["NORMAL", "CAUTION", "WARNING"]
    message: str = Field(min_length=1, max_length=200)
    voice_message: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=180)
    recommended_action: str = Field(min_length=1, max_length=120)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)
    suggested_sequence: list[str] = Field(default_factory=list, max_length=3)


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
        self._last_level = "ok"
        self._last_key = None
        self._last_discomfort_id = None
        self.llm_errors = 0
        self.last_latency_ms: int | None = None
        self.service_status = "unverified" if client else "offline"
        self.cache = {}

        self._request_times = deque()
        self._budget_lock = threading.Lock()
        self._retry_after = 0.
        self._failures = 0
        self._thread = threading.Thread(target=self._worker, name="coach", daemon=True)
        self._thread.start()

    @property
    def provider(self) -> str:
        return self.client.name if self.client else "rules"

    # called by the pipeline at ~5 Hz
    def observe(self, state: dict, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        a = state.get("attention", {})
        if state.get("status", "running") != "running" or not a.get("calibrated", True):
            return
        level = a.get("level", "ok")
        loc = local_message(state)
        key = loc["reason"] if loc else None

        # 1) instant local alert on a new danger reason (repeat every 6 s while it lasts)
        if loc and loc["severity"] == "alert":
            if now - self._last_local.get(key, -1e9) > 6:
                self._last_local[key] = now
                self._emit({**loc, "source": "instant"})
        elif loc and key != self._last_key and key in ("both hands off the wheel", "head dropping", "driver drowsiness"):
            self._emit({**loc, "source": "instant"})

        # 2) LLM on change, or periodic summary
        changed = key != self._last_key or level != self._last_level
        due = now - self._last_llm >= (self.min_interval_s if changed and level != "ok" else self.summary_every_s)
        self._last_key, self._last_level = key, level
        event = (state.get("discomfort") or {}).get("event")
        new_event = event and event["id"] != self._last_discomfort_id and now-self._last_llm >= self.min_interval_s
        if new_event:
            self._last_discomfort_id = event["id"]
            if loc is None:
                loc = {"message": "Are you comfortable? If needed, take a break when safe.",
                       "severity": "info", "reason": "facial movement"}
            due = True
        if due and (changed or new_event or self.summary_every_s == 0):
            self._last_llm = now
            self._submit(("react", compact_state(state), loc))

    def explain(self, alert, snapshot):
        events = snapshot["events"]
        occurrence = next((o for o in reversed(snapshot["occurrences"]) if o["pattern_id"] == alert["pattern_id"]), None)
        if occurrence is None:
            return None
        selected = set(occurrence["supporting_event_ids"])
        evidence = [{k:e[k] for k in ("id","type","duration")} for e in events if e["id"] in selected]
        if len(evidence) != len(selected):
            return None
        compact = dict(pattern_name=alert["name"], severity=alert["severity"], evidence=evidence,
                       history=[e["type"] for e in events[-16:]])
        self._submit(("pattern", compact, dict(message=alert["message"], reason=alert["name"], severity="info", alert_id=alert["id"])))
        return self.cache.get(self._cache_key(compact), {}).get("voice_message")

    def _complete(self, system, prompt, json_mode=False, schema=None):
        now = time.monotonic()
        with self._budget_lock:
            while self._request_times and now-self._request_times[0] >= 3600:
                self._request_times.popleft()
            if now < self._retry_after or len(self._request_times) >= 20:
                if len(self._request_times) >= 20:
                    self.service_status = "local request budget exhausted"
                raise RuntimeError("local backoff")
            self._request_times.append(now)
        try:
            options = {"schema": schema} if schema and isinstance(self.client, GeminiClient) else {}
            result = self.client.complete(system, prompt, json_mode=json_mode, **options)
        except Exception as exc:
            self._failures += 1
            self._retry_after = now + min(300, 15 * 2 ** min(self._failures, 5))
            self.service_status = f"unavailable ({getattr(exc, 'code', None) or type(exc).__name__})"
            raise
        self._failures = 0
        self.service_status = "online"
        return result

    @staticmethod
    def _cache_key(compact):
        return (compact["pattern_name"], compact["severity"], tuple(compact["history"]),
                tuple((e["type"], e["duration"]) for e in compact["evidence"]))

    def _pattern_reply(self, compact):
        key = self._cache_key(compact)
        if key in self.cache:
            return self.cache[key]
        prompt = ("Explain only the referenced driver-facing observations. No claims about road conditions, "
                  "objects, diagnoses or measurements beyond evidence. Provide calm guidance in "
                  + self.language + " in at most 15 words. Copy pattern_name exactly. Reference evidence_ids. "
                  "Optionally suggest a sequence of observed event types. Return JSON matching "
                  + json.dumps(PatternReply.model_json_schema()))
        reply = PatternReply.model_validate_json(
            self._complete(prompt, json.dumps(compact), True, schema=PatternReply.model_json_schema()))
        if (reply.pattern_name != compact["pattern_name"] or len(reply.voice_message.split()) > 15
                or not set(reply.evidence_ids) <= {e["id"] for e in compact["evidence"]}
                or not set(reply.suggested_sequence) <= set(compact["history"])):
            raise ValueError("unsupported pattern reply")
        explanation = " ".join((reply.message, reply.reason, reply.voice_message, reply.recommended_action))
        if re.search(r"\b(pedestrians?|traffic lights?|crosswalks?|speed limit|oncoming|cyclists?|potholes?)\b",
                     explanation, re.IGNORECASE):
            raise ValueError("outside driver-camera evidence")
        observed = {float(len(compact["evidence"])), *(float(e["duration"]) for e in compact["evidence"])}
        if any(float(n) not in observed for n in re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", explanation)):
            raise ValueError("unsupported measurement")
        out = dict(message=reply.message, reason=reply.reason, voice_message=reply.voice_message,
                   suggested_sequence=reply.suggested_sequence, pattern_name=reply.pattern_name, severity="info")
        if len(self.cache) >= 64:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out

    def stop(self):
        self._jobs.put(None)
        self._thread.join()

    def _submit(self, job) -> None:
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            if job[0] == "pattern":
                try:
                    self._jobs.get_nowait()
                    self._jobs.put_nowait(job)
                except (queue.Empty, queue.Full):
                    pass

    def chat(self, text: str, state: dict) -> dict:
        """Synchronous Q&A for the dashboard's 'talk to the co-pilot' box."""
        msg = {"role": "user", "message": text[:500], "severity": "info", "source": "user", "t": time.time()}
        self.history.append(msg)
        if not self.client:
            reply = "I'm in offline mode (no LLM key). " + (local_message(state) or {"message": "All looks fine — drive safe."})["message"]
        else:
            try:
                reply = self._complete(CHAT_PROMPT.format(language=self.language),
                                             f"Live state: {json.dumps(compact_state(state))}\n\nDriver/passenger says: {text}")
            except Exception as e:
                log.warning("chat failed (%s)", type(e).__name__)
                self.llm_errors += 1
                reply = "Sorry, I can't reach the assistant right now."
        out = {"message": reply.strip()[:400], "severity": "info", "reason": "chat", "source": self.provider if self.service_status == "online" else "rules"}
        self._emit(out)
        return out

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, compact, loc = job
            out = None
            if self.client:
                t0 = time.monotonic()
                try:
                    if kind == "pattern":
                        out = self._pattern_reply(compact)
                    else:
                        txt = self._complete(SYSTEM_PROMPT.format(language=self.language),
                                             "Driver state now:\n" + json.dumps(compact), json_mode=True)
                        out = parse_reply(txt)
                    self.last_latency_ms = int((time.monotonic() - t0) * 1000)
                except Exception as e:
                    self.llm_errors += 1
                    log.warning("LLM response unavailable (%s)", type(e).__name__)
                    if self.service_status == "online":
                        self.service_status = "invalid response"
            if out is None:
                out = loc or {"message": "All good. Keep your eyes on the road.", "severity": "info", "reason": "ok"}
                out = {**out, "source": "rules"}
            else:
                out["source"] = self.provider
            if kind == "pattern" and loc:
                out = {**out, "alert_id":loc["alert_id"]}
            self._emit(out)

    def _emit(self, msg: dict) -> None:
        msg = {**msg, "t": time.time(), "id": uuid4().hex, "timestamp":datetime.now(timezone.utc).isoformat()}
        self.history.append(msg)
        try:
            self.on_message(msg)
        except Exception as e:
            log.warning("on_message failed: %s", e)
