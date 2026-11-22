"""Keep the detector loop fast when the laptop is busy.

Measured on an i7-13650HX (6 P + 8 E cores) with other apps using ~7 cores: YOLOv8n at 640 px
took 7.6 s per frame with torch's default 14 threads (OpenMP threads spin-waiting on each other
while the OS pre-empts them) and 0.13 s with 4. That is the "0.8 proc fps" collapse.
Windows can also move a background process to the slow E-cores (power throttling / EcoQoS).
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform

log = logging.getLogger("perf")

TORCH_THREADS = int(os.getenv("TORCH_THREADS", "4"))
CV2_THREADS = int(os.getenv("CV2_THREADS", "2"))
# OpenMP reads this when torch is imported; set_num_threads() alone only reaches the calling thread
os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_THREADS))


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def _windows_no_throttling() -> None:
    k32 = ctypes.windll.kernel32
    proc = k32.GetCurrentProcess()
    # ProcessPowerThrottling = 4; control EXECUTION_SPEED (1) with state 0 = never throttle
    st = _PowerThrottlingState(1, 0x1, 0x0)
    if not k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st)):
        log.warning("could not turn off Windows power throttling (error %d)", k32.GetLastError())
    ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
    if not k32.SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS):
        log.warning("could not raise process priority (error %d)", k32.GetLastError())


def limit_threads() -> None:
    """Call on every thread that runs torch (the OpenMP thread count is per calling thread)."""
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except Exception:
        pass


def tune() -> None:
    """Call once before the detectors load. Never raises."""
    if platform.system() == "Windows":
        try:
            _windows_no_throttling()
        except Exception:
            log.exception("Windows performance tuning failed")
    try:
        import cv2
        cv2.setNumThreads(CV2_THREADS)
    except Exception:
        pass
    limit_threads()
