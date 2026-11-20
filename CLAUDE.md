# AI_FastGears — guidance for Claude Code

Hackathon project (Codelap 2026, "AI Drive"): live AI driver-monitoring assistant.
Deadline-driven: small, working, demoable increments; one commit per feature.

## Judging requirements (map every change to one)
- **R1 Driver detection (CV):** eyes closed/drowsiness, looking left/right/down, phone, posture.
- **R2 In-car reaction:** alert / warning / dialogue / voice — an LLM is required.
- **R3 Live demo:** live camera feed only, no pre-recorded or labelled videos.
- **R4 Live dashboard:** all detections + live text/audio suggestion in real time.
- Deliverables: README (idea, architecture, how to run), presentation in `docs/presentation/AI_FastGears.pptx`.

## Architecture
`python -m fastgears` -> `server.py` (FastAPI, `/ws`, `/video.mjpg`) -> `pipeline.py` (worker thread)
-> `vision/face.py` (MediaPipe, all faces) -> `vision/driver.py` (DriverLock: only the driver is measured) -> `vision/rppg.py` (camera pulse) + `vision/objects.py` (YOLOv8n phone)
-> `vision/attention.py` (level ok/caution/danger + reasons) -> `llm/coach.py` (instant alert + Gemini) -> `web/index.html`.

## Rules
- No special hardware: laptop CPU + webcam; LLM via API (Gemini default, Anthropic optional, rules fallback).
- `vision/rppg.py`, `face.py` (`_euler_from_matrix`), `filters.py`, `identity.py` are ported from the validated
  afferent-junction2026 repo. Don't change `pos_spectrum` / `trust_score` maths without a test.
- Detectors are optional: a failing model must not crash the pipeline.
- `attention.py` is time-driven (`t` argument, never wall clock) so it stays unit-testable.
- The LLM never blocks the video loop; danger alerts never wait for the LLM.
- Secrets only in `.env` (gitignored).
- Tests: `.venv\Scripts\python -m pytest -q` (no camera or models needed).
