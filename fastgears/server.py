"""FastAPI app: live dashboard (R4), MJPEG camera stream, WebSocket state + co-pilot messages."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .llm import Coach, make_client
from .llm.intervention import CheckIn
from .pipeline import Pipeline

log = logging.getLogger("server")
WEB = Path(__file__).parent / "web"


class Hub:
    """Fan-out of messages to every connected dashboard."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish_threadsafe(self, msg: dict) -> None:
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, default=_json_default)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                self.clients.discard(ws)


def _json_default(x):
    """numpy scalars / arrays that slipped into the state: send them as plain numbers / lists."""
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


hub = Hub()
pipeline: Pipeline | None = None
coach: Coach | None = None
voice_on = True   # one Voice on/off switch for every open page (dashboard, co-pilot, any host name)


def _on_state(state: dict) -> None:
    if coach:
        coach.observe(state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, coach
    hub.loop = asyncio.get_running_loop()
    coach = Coach(make_client(settings),
                  on_message=lambda m: hub.publish_threadsafe({"type": "message", **m}),
                  language=settings.language,
                  checkin=CheckIn(settings.checkin_answer_s, settings.checkin_max_rounds,
                                  settings.checkin_cooldown_s) if settings.checkin else None)
    try:
        coach.set_tuning(settings.message_preset)
    except ValueError:
        log.warning("MESSAGE_PRESET=%s unknown: using normal", settings.message_preset)
        coach.set_tuning("normal")
    pipeline = Pipeline(settings, on_state=_on_state).start()
    task = asyncio.create_task(_state_pump())
    log.info("dashboard: http://%s:%d  (LLM: %s)", settings.host, settings.port, coach.provider)
    yield
    task.cancel()
    pipeline.stop()


async def _state_pump():
    """Push the latest state to dashboards at 5 Hz."""
    while True:
        await asyncio.sleep(0.2)
        try:
            await _push_state()
        except Exception:   # one bad state must not stop the dashboard for good
            log.exception("state push failed")


async def _push_state():
    if pipeline and hub.clients:
        st = pipeline.state()
        st["llm"] = {"provider": coach.provider, "errors": coach.llm_errors,
                     "latency_ms": coach.last_latency_ms, "stt": coach.can_transcribe} if coach else None
        st["engagement"] = coach.engagement.status() if coach else {"replies": 0}
        st["awaiting_reply"] = coach.awaiting_reply(coach._t) if coach else False
        st["check"] = coach.checkin_status() if coach else {"active": False}
        st["voice"] = voice_on
        await hub.broadcast({"type": "state", **st})


app = FastAPI(title="FastGears driver monitor", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/copilot")
def copilot():
    """Driver-facing screen: the co-pilot's message and voice, no technical data."""
    return FileResponse(WEB / "copilot.html")


@app.get("/api/health")
def health():
    return {"ok": True, "camera": pipeline.camera.connected if pipeline else False,
            "llm": coach.provider if coach else None}


@app.get("/api/state")
def state():
    return JSONResponse(pipeline.state() if pipeline else {})


@app.get("/api/messages")
def messages():
    return list(coach.history) if coach else []


@app.post("/api/recalibrate")
def recalibrate():
    pipeline.recalibrate()
    return {"ok": True}


class TuningIn(BaseModel):
    preset: str | None = None
    message_gap_s: float | None = None
    warn_hold_s: float | None = None
    alert_repeat_s: float | None = None
    question_cooldown_s: float | None = None
    questions: bool | None = None


@app.get("/api/tuning")
def get_tuning():
    return coach.tuning()


@app.post("/api/tuning")
def set_tuning(body: TuningIn):
    try:
        return coach.set_tuning(**body.model_dump())
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


class VoiceIn(BaseModel):
    on: bool


@app.get("/api/voice")
def get_voice():
    return {"on": voice_on}


@app.post("/api/voice")
async def set_voice(body: VoiceIn):
    global voice_on
    voice_on = body.on
    await hub.broadcast({"type": "voice", "on": voice_on})   # every page follows at once
    return {"on": voice_on}


class ChatIn(BaseModel):
    text: str
    voice: dict | None = None            # how the reply sounded (from /api/transcribe)
    reply_time_s: float | None = None    # question end -> first word, measured in the browser


@app.post("/api/chat")
async def chat(body: ChatIn):
    st = pipeline.state() if pipeline else {}
    return await asyncio.to_thread(coach.chat, body.text, st, body.voice, body.reply_time_s)


class AnsweringIn(BaseModel):
    active: bool


@app.post("/api/answering")
def answering(body: AnsweringIn):
    """Browser: the driver started speaking a reply (True) or gave up (False). While a reply is
    on its way the co-pilot keeps measuring but stays silent, and the question doesn't time out."""
    coach.set_answering(body.active)
    return {"ok": True}


MAX_AUDIO_BYTES = 2_000_000   # ~60 s of 16 kHz mono WAV


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """The driver's spoken reply (raw audio body, e.g. audio/wav) -> transcript + voice cues, via Gemini."""
    if not coach or not coach.can_transcribe:
        return JSONResponse({"error": "speech-to-text needs LLM_PROVIDER=gemini and GEMINI_API_KEY"}, status_code=503)
    audio = await request.body()
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        return JSONResponse({"error": "empty or too long recording"}, status_code=400)
    mime = request.headers.get("content-type", "audio/wav").split(";")[0]
    try:
        return await asyncio.to_thread(coach.transcribe, audio, mime)
    except Exception as e:
        log.warning("transcribe failed: %s", str(e)[:200])
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return JSONResponse({"error": "Gemini quota used up", "quota": True}, status_code=429)
        return JSONResponse({"error": "transcription failed"}, status_code=502)


@app.get("/video.mjpg")
async def video():
    async def gen():
        last = None
        while True:
            jpg = pipeline.jpeg() if pipeline else None
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1 / 25)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        for m in list(coach.history)[-10:] if coach else []:
            await websocket.send_text(json.dumps({"type": "message", **m}))
        while True:
            await websocket.receive_text()   # keepalive; client sends nothing important
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)
