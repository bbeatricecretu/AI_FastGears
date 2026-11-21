"""Settings, read once from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # optional: python-dotenv
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _camera(v: str):
    """'0' -> 0 (webcam index); anything else is a path/URL (dev replay only)."""
    return int(v) if v.strip().isdigit() else v


@dataclass
class Settings:
    camera: int | str = field(default_factory=lambda: _camera(os.getenv("CAMERA", "0")))
    camera_backend: str = os.getenv("CAMERA_BACKEND", "auto").strip().lower()
    width: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    height: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    mirror: bool = _bool("MIRROR", True)

    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    models_dir: str = os.getenv("MODELS_DIR", str(ROOT / "models"))
    data_dir: str = os.getenv("DATA_DIR", str(ROOT / "data"))

    # detectors
    enable_rppg: bool = _bool("ENABLE_RPPG", True)
    enable_objects: bool = _bool("ENABLE_OBJECTS", True)
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    yolo_every: int = int(os.getenv("YOLO_EVERY", "3"))
    driver_seat: str = os.getenv("DRIVER_SEAT", "left").lower()   # auto | left | right (as on screen)
    enable_hands: bool = _bool("ENABLE_HANDS", True)
    calib_s: float = float(os.getenv("CALIBRATION_S", "5"))
    discomfort_overlap_s: float = float(os.getenv("DISCOMFORT_OVERLAP_S", "0.3"))
    enable_rgb: bool = _bool("ENABLE_RGB", False)
    rgb_reference: str = os.getenv("RGB_REFERENCE", "")
    rgb_overlay: bool = _bool("RGB_OVERLAY", False)
    rgb_calibration_s: str = os.getenv("RGB_CALIBRATION_S", "5")
    rgb_persistence_s: str = os.getenv("RGB_PERSISTENCE_S", "1")

    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()   # gemini | anthropic | none
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    phone_vision_check: bool = _bool("PHONE_VISION_CHECK", True)
    gemini_vision_model: str = os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")
    llm_min_interval_s: float = float(os.getenv("LLM_MIN_INTERVAL_S", "8"))
    llm_summary_every_s: float = float(os.getenv("LLM_SUMMARY_EVERY_S", "45"))
    language: str = os.getenv("ASSISTANT_LANGUAGE", "English")
    # closed-loop check-in after a danger alert: question -> answer + camera reaction -> verdict
    checkin: bool = _bool("CHECKIN", True)
    checkin_answer_s: float = float(os.getenv("CHECKIN_ANSWER_S", "12"))
    checkin_max_rounds: int = int(os.getenv("CHECKIN_MAX_ROUNDS", "3"))
    checkin_cooldown_s: float = float(os.getenv("CHECKIN_COOLDOWN_S", "30"))


settings = Settings()
