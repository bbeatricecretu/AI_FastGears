"""FastAPI app: live dashboard (R4), MJPEG camera stream, WebSocket state + co-pilot messages."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .llm import Coach, make_client
from .pipeline import Pipeline
from .history import History
from .monitor import Monitor, Thresholds, utc, uid
from urllib.parse import urlsplit

log = logging.getLogger("server")
WEB = Path(__file__).parent / "web"


class Hub:
    """Fan-out of messages to every connected dashboard."""

    def __init__(self):
        self.clients: dict[WebSocket, asyncio.Queue] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish_threadsafe(self, msg: dict) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self.broadcast, msg.copy())

    def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg)
        for websocket, pending in list(self.clients.items()):
            if msg["type"] == "state" and pending.qsize() > 2:
                continue
            if pending.full():
                self.clients.pop(websocket, None)
                asyncio.create_task(websocket.close(code=1013))
            else:
                pending.put_nowait(data)

    async def send(self, websocket, pending):
        while True:
            await websocket.send_text(await pending.get())


hub = Hub()
pipeline: Pipeline | None = None
coach: Coach | None = None


monitor: Monitor | None = None


def _publish_alert(message):
    if coach and monitor and message.get("pattern_id"):
        voice = coach.explain(message, monitor.snapshot())
        if message["type"] == "alert" and coach.client and coach.service_status in ("online", "unverified"):
            message = {**message, "await_voice":True}
        if voice:
            message = {**message, "voice_message":voice}
    hub.publish_threadsafe(message)


def _on_message(message):
    sequence = message.get("suggested_sequence")
    if sequence and monitor:
        message = {**message, "suggestion_validated":monitor.validate_suggestion(sequence)}
    hub.publish_threadsafe({"type":"message", **message})


def _on_state(state: dict) -> None:
    if monitor:
        monitor.observe(state)
    if coach:
        coach.observe(state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, coach, monitor
    hub.loop = asyncio.get_running_loop()
    coach = Coach(make_client(settings),
                  on_message=_on_message,
                  min_interval_s=settings.llm_min_interval_s,
                  summary_every_s=settings.llm_summary_every_s,
                  language=settings.language)
    db_name = "synthetic.sqlite3" if settings.synthetic else "driver.sqlite3"
    monitor = Monitor(History(Path(settings.data_dir).resolve() / db_name), _publish_alert,
                      Thresholds(settings.pattern_window, settings.pattern_minimum, settings.pattern_recovery,
                                 settings.pattern_count, settings.pattern_cooldown, settings.pattern_support,
                                 settings.pattern_growth, settings.alert_hold))
    factory = Pipeline
    if settings.synthetic:
        from .synthetic import DevelopmentPipeline
        factory = DevelopmentPipeline
    pipeline = factory(settings, on_state=_on_state).start()
    task = asyncio.create_task(_state_pump())
    log.info("dashboard: http://%s:%d  (LLM: %s)", settings.host, settings.port, coach.provider)
    try:
        yield
    finally:
        task.cancel()
        pipeline.stop()
        coach.stop()
        monitor.close()
        hub.loop = None
        monitor = pipeline = coach = None


async def _state_pump():
    """Push the latest state to dashboards at 5 Hz."""
    while True:
        await asyncio.sleep(0.2)
        if pipeline and hub.clients:
            st = pipeline.state()
            st["monitor"] = monitor.snapshot()
            st["synthetic"] = settings.synthetic
            st["llm"] = {"provider": coach.provider, "errors": coach.llm_errors,
                         "status": coach.service_status, "latency_ms": coach.last_latency_ms} if coach else None
            hub.broadcast({"type":"state", "id":uid(), "timestamp":utc(), **st})


app = FastAPI(title="FastGears driver monitor", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


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


class ChatIn(BaseModel):
    text: str


@app.post("/api/chat")
async def chat(body: ChatIn):
    st = pipeline.state() if pipeline else {}
    return await asyncio.to_thread(coach.chat, body.text, st)


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
    origin = websocket.headers.get("origin")
    if origin and urlsplit(origin).netloc != websocket.headers.get("host"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    pending = asyncio.Queue(maxsize=64)
    hub.clients[websocket] = pending
    await websocket.send_json({"type":"snapshot", "id":uid(), "timestamp":utc(),
                               "state": pipeline.state() if pipeline else {},
                               "monitor": monitor.snapshot() if monitor else {},
                               "messages": list(coach.history)[-10:] if coach else []})
    sender = asyncio.create_task(hub.send(websocket, pending))
    try:
        while True:
            await websocket.receive_text()   # keepalive; client sends nothing important
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.pop(websocket, None)
        sender.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await sender

@app.get("/api/patterns")
def patterns(limit: int = 50):
    return monitor.history.recent("patterns", limit) if monitor else []


@app.get("/api/patterns/recent")
def occurrences(limit: int = 50):
    return monitor.history.recent("pattern_occurrences", limit) if monitor else []


@app.get("/api/events/recent")
def events(limit: int = 50):
    return monitor.history.recent("events", limit) if monitor else []


@app.get("/api/alerts")
def alerts(limit: int = 50):
    return monitor.history.recent("alerts", limit) if monitor else []


@app.get("/api/alerts/active")
def active():
    item = monitor.snapshot()["active"] if monitor else None
    return [item] if item else []


@app.get("/api/session/stats")
def stats():
    return monitor.snapshot() if monitor else {}


@app.get("/api/system/status")
def system_status():
    return {**health(), "llm_status": coach.service_status if coach else "offline",
            "storage": monitor.history.error or "ready" if monitor else "offline",
            "synthetic": settings.synthetic}
