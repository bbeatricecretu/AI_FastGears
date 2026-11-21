"""Explicit development input for the same detector-state callback used by Pipeline."""
import threading
import time
from types import SimpleNamespace


class DevelopmentPipeline:
    def __init__(self, settings, on_state):
        self.callback = on_state
        self.camera = SimpleNamespace(connected=False)
        self.done = threading.Event()
        self.latest = {"status":"starting", "synthetic":True}
        self.worker = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.worker.start()
        return self

    def run(self):
        start = time.monotonic()
        while not self.done.is_set():
            t = time.monotonic()-start
            phase = t % 24
            looking = "left" if any(b <= phase < b+1 for b in (2, 6, 10)) else "road"
            self.latest = dict(status="running", synthetic=True, uptime_s=t, camera={"connected":False},
                               attention=dict(face_present=True, calibrated=True, looking=looking, level="ok",
                                              yawning=14 <= phase < 15, eyes_closed=17 <= phase < 18))
            self.callback(self.latest)
            self.done.wait(.1)

    def state(self):
        return self.latest.copy()

    def jpeg(self):
        return None

    def recalibrate(self):
        pass

    def stop(self):
        self.done.set()
        self.worker.join()
