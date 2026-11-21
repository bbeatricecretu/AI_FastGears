import threading

from fastapi.testclient import TestClient

from fastgears import server
from fastgears.history import History
from fastgears.monitor import Monitor, Thresholds


def frame(looking="road", **fields):
    return {"status":"running", "attention":{"calibrated":True, "face_present":True,
            "level":"ok", "looking":looking, **fields}}


def episode(feed, t, state):
    feed(state, t)
    feed(state, t + .6)
    feed(frame(), t + 1)


def test_distraction_delivery_and_reconnect(tmp_path, monkeypatch):
    class Camera:
        connected = True

    class ManualPipeline:
        camera = Camera()
        def __init__(self, settings, on_state):
            self.feed = on_state
        def start(self):
            return self
        def state(self):
            return frame()
        def stop(self):
            pass

    monkeypatch.setattr(server, "Pipeline", ManualPipeline)
    monkeypatch.setattr(server.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(server.settings, "synthetic", False)
    monkeypatch.setattr(server.settings, "llm_provider", "none")
    clock = [0.]
    real = Monitor.observe
    monkeypatch.setattr(Monitor, "observe", lambda self, state: real(self, state, clock[0]))
    def feed(state, t):
        clock[0] = t
        server.pipeline.feed(state)
    def receive_alert(ws):
        for _ in range(50):
            m = ws.receive_json()
            if m["type"] == "alert":
                return m
        raise AssertionError("missing alert")
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            assert a.receive_json()["type"] == b.receive_json()["type"] == "snapshot"
            for t in (1,2,3,4,6):
                feed(frame("left"), t)
            feed(frame(), 6.5)
            assert not server.monitor.alerts
            assert not server.monitor.occurrences
            episode(feed, 9, frame("right"))
            assert not server.monitor.alerts
            episode(feed, 14, frame("left"))
            alert = receive_alert(a)
            assert alert == receive_alert(b)
            assert alert["severity"] == "CAUTION" and alert["name"] == "Repeated distraction"
        with client.websocket_connect("/ws") as ws:
            snapshot = ws.receive_json()
            assert snapshot["monitor"]["active"]["id"] == alert["id"]
        server.monitor.history.pending.join()
        rows = client.get("/api/events/recent").json()
        assert len(rows) == 6
        assert sorted(e["duration"] for e in rows if e["type"]=="LOOKING_LEFT") == [.6, 5]
        matches = client.get("/api/patterns/recent").json()
        assert len(matches) == 1 and len(matches[0]["supporting_event_ids"]) == 3
        assert len(client.get("/api/alerts").json()) == 1
        assert client.get("/api/system/status").json()["llm_status"] == "offline"


def test_historical_discovery_reload_and_negative_control(tmp_path):
    path = tmp_path/"discovery.sqlite3"
    monitor = Monitor(History(path), lambda msg: None)
    for t in (0,40,80):
        episode(monitor.observe,t,frame(yawning=True))
        episode(monitor.observe,t+3,frame(phone=True))
    monitor.close()
    monitor = Monitor(History(path), lambda msg: None)
    pattern = next(p for p in monitor.patterns.values() if p["conditions"].get("mode")=="sequence"
                   and p["conditions"]["types"]==["YAWNING","PHONE_DETECTED"])
    assert pattern["support"] == 3
    episode(monitor.observe,1,frame(phone=True))
    assert monitor.patterns[pattern["id"]]["support"] == 3
    episode(monitor.observe,50,frame(yawning=True))
    episode(monitor.observe,54,frame(phone=True))
    monitor.close()
    saved = [o for o in monitor.history.recent("pattern_occurrences",512) if o["pattern_id"]==pattern["id"]]
    assert len(saved)==4 and len({i for o in saved for i in o["supporting_event_ids"]})==8


def test_discovery_from_saved_history_before_next_live_episode(tmp_path):
    path = tmp_path / "history.sqlite3"
    training = Monitor(History(path), lambda _: None, Thresholds(support=4))
    for t in (0, 40, 80):
        episode(training.observe, t, frame(yawning=True))
        episode(training.observe, t + 3, frame(phone=True))
    training.close()
    assert not training.history.recent("patterns")

    deliveries = []
    monitor = Monitor(History(path), deliveries.append, Thresholds(support=3))
    try:
        sequence = next(p for p in monitor.patterns.values()
                        if p["conditions"].get("mode") == "sequence"
                        and p["conditions"]["types"] == ["YAWNING", "PHONE_DETECTED"])
        assert sequence["support"] == 3
        assert not deliveries
        episode(monitor.observe, 1, frame(phone=True))
        assert monitor.patterns[sequence["id"]]["support"] == 3
        episode(monitor.observe, 50, frame(yawning=True))
        episode(monitor.observe, 54, frame(phone=True))
    finally:
        monitor.close()
    matches = [o for o in monitor.history.recent("pattern_occurrences", 512)
               if o["pattern_id"] == sequence["id"]]
    assert len(matches) == 4
    assert len({o["session_id"] for o in matches}) == 2


def test_three_gazes_outside_window(tmp_path):
    monitor = Monitor(History(tmp_path / "window.sqlite3"), lambda _: None)
    try:
        for t in (0, 2, 32):
            episode(monitor.observe, t, frame("left"))
        assert not any(a["name"] == "Repeated distraction" for a in monitor.alerts)
    finally:
        monitor.close()


def test_critical_immediate_slow_storage_and_camera_loss(tmp_path):
    messages = []
    monitor = Monitor(History(tmp_path/"critical.db"), messages.append)
    monitor.history.pending.join()
    gate = threading.Event()
    # An independent SQLite writer holds the write lock while the detector runs.
    import sqlite3
    db = sqlite3.connect(monitor.history.path)
    db.execute("BEGIN IMMEDIATE")
    def detect():
        monitor.observe(frame(microsleep=True,eyes_closed=True,level="danger"),0)
        monitor.observe(frame(microsleep=True,eyes_closed=True,level="danger"),1)
        gate.set()
    worker = threading.Thread(target=detect)
    worker.start()
    try:
        assert gate.wait(1)
        assert [m["severity"] for m in messages] == ["WARNING"]
    finally:
        db.rollback()
        db.close()
        worker.join()
    monitor.observe({"status":"no camera"},2)
    monitor.close()
    assert monitor.history.recent("events")[0]["status"]=="interrupted"


def test_escalation_priority_recovery_and_cooldown(tmp_path):
    received = []
    monitor = Monitor(History(tmp_path/"priority.db"), received.append)
    for t, duration in [(0,.5),(4,1),(8,1.5)]:
        monitor.observe(frame("left"), t)
        monitor.observe(frame("right", distracted=True), t+duration)
        monitor.observe(frame(), t+duration+.35)
    alerts = [m for m in received if m["type"] == "alert"]
    assert alerts[-1]["name"] == "Escalating inattention" and alerts[-1]["severity"]=="WARNING"
    assert len([m for m in alerts if m["pattern_id"]]) == 1
    before = monitor.counts["CAUTION"]
    episode(monitor.observe, 11, frame("down"))
    assert monitor.counts["CAUTION"] == before
    assert len([o for o in monitor.occurrences if o["name"]=="Repeated distraction"]) == 1
    monitor.close()


def test_fatigue_phone_and_unavailable_inputs(tmp_path):
    received = []
    monitor = Monitor(History(tmp_path/"signals.db"), received.append)
    episode(monitor.observe,0,frame("left"))
    episode(monitor.observe,3,frame("down",phone=True))
    for t in (7,11,15):
        episode(monitor.observe,t,frame(yawning=True))
    assert {"Phone-related distraction","Repeated fatigue indicators","Fatigue and distraction"} <= {
        o["name"] for o in monitor.occurrences}
    episode(monitor.observe,40,frame("left",face_present=False))
    episode(monitor.observe,44,frame("left",calibrated=False))
    assert not any(e["start"]>=40 for e in monitor.events)
    monitor.close()


def test_gemini_suggestion_requires_independent_support(tmp_path):
    monitor = Monitor(History(tmp_path/"suggestion.db"),lambda m:None)
    types = ["YAWNING","LOOKING_LEFT","EYES_CLOSED"]
    for t in (0,40,80):
        episode(monitor.observe,t,frame(yawning=True))
        episode(monitor.observe,t+3,frame("left"))
        episode(monitor.observe,t+6,frame(eyes_closed=True))
        monitor.history.pending.join()
        assert monitor.validate_suggestion(types) == (t==80)
    assert not monitor.validate_suggestion(["arbitrary code", "YAWNING"])
    learned = next(p for p in monitor.patterns.values() if p["source"]=="gemini-suggested")
    assert learned["support"] == 3
    monitor.close()


def test_storage_unavailable_keeps_critical_alerts(tmp_path):
    invalid = tmp_path/"file"
    invalid.write_text("file")
    got=[]
    monitor=Monitor(History(invalid/"journal.db"),got.append)
    monitor.observe(frame(microsleep=True,level="danger"),1)
    assert got[0]["severity"]=="WARNING"
    assert monitor.snapshot()["storage_error"]
    monitor.close()
