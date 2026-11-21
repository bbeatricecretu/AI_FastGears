"""Webcam capture on its own thread: always hands out the newest frame, never a backlog."""
from __future__ import annotations

import logging
import platform
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("camera")


class Camera:
    def __init__(self, source: int | str = 0, width: int = 1280, height: int = 720,
                 mirror: bool = True, backend: str = "auto"):
        self.source, self.width, self.height, self.mirror = source, width, height, mirror
        if backend not in ("auto", "dshow", "msmf"):
            raise ValueError(f"Invalid camera backend: {backend}")
        self.backend = backend
        self._frame: np.ndarray | None = None
        self._frame_t: float | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.connected = False
        self.fps = 0.0
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)

    def start(self) -> "Camera":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture:
        if self.backend != "auto" and isinstance(self.source, int):
            api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[self.backend]
            cap = cv2.VideoCapture(self.source, api)
        elif isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)   # opens much faster than MSMF
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        is_file = not isinstance(self.source, int)
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                log.warning("camera %r not available, retrying", self.source)
                time.sleep(2)
                continue
            self.connected = True
            log.info("camera %r open via %s at %dx%d", self.source, cap.getBackendName(),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            file_dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if is_file else 0
            n, t0 = 0, time.monotonic()
            while not self._stop.is_set():
                ok, frame = cap.read()
                captured_at = time.monotonic()
                if not ok:
                    if is_file:                       # loop a replay file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                with self._cond:
                    self._frame, self._seq = frame, self._seq + 1
                    self._frame_t = captured_at
                    self._cond.notify_all()
                n += 1
                if time.monotonic() - t0 >= 2:
                    self.fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
                if file_dt:
                    time.sleep(file_dt)
            cap.release()
            self.connected = False
            log.warning("camera lost, reopening")
            time.sleep(1)

    def read(self, last_seq: int = -1, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than last_seq arrives; returns (seq, frame)."""
        seq, frame, _ = self.read_timed(last_seq, timeout)
        return seq, frame

    def read_timed(self, last_seq: int = -1, timeout: float = 1.0) -> tuple[int, np.ndarray | None, float | None]:
        """Return the latest frame with its monotonic arrival time at the capture thread."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self._stop.is_set(), timeout=timeout)
            return self._seq, self._frame, self._frame_t
