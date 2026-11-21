# Bug burn-down — 2026-11-22

Scope: the Python backend (`fastgears/`) and the WebSocket and network code of the two web pages.
Every fix has a regression test in `tests/test_burn_down.py`. Each test was run against the old
code (it failed) and against the fixed code (it passes).

**Result:** 0 Critical · 0 High · 0 Medium open. Full suite: **231 passed** (`.venv\Scripts\python -m pytest -q`).

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | A stalled LLM call during a check-in verdict mutes the co-pilot (all alerts except microsleep) indefinitely | High | Fixed |
| 2 | The Gemini vision phone check stops for good after one stalled request | Medium | Fixed |
| 3 | A single non-JSON value in the state permanently stops the 5 Hz dashboard push | Low | Fixed (hardening) |
| 4 | With the camera gone, the dashboard keeps showing the last frame's hands and phone detections | Low | Fixed |
| 5 | Each WebSocket reconnect adds another 15 s ping timer (timer leak) | Low | Fixed |

## 1 — Hanging check-in judge mutes danger alerts (High, R2)
- **Repro:** `test_bug1_…`: an LLM whose `complete()` never returns. The driver answers the
  check-in question, and `CheckIn` stays in phase `judging` forever. `Coach.observe` treats any active
  check-in as busy, so later dangers (e.g. 10 s of phone use) are never spoken.
- **Root cause:** `Coach._judge_async` waited on the LLM with no deadline, and the Gemini and Anthropic
  clients had no network timeout (the google-genai default is none). If the judge thread raised an
  exception, the round also stayed in `judging`. This broke the rule that danger alerts never wait for the LLM.
- **Fix:** `coach.py`: the verdict waits at most `JUDGE_WAIT_S` (6 s) for the LLM; after that
  the camera rules decide (`rule_verdict`). Whichever answer comes first is applied and the other
  is ignored. A crash in the judge also falls back to the rules. All LLM clients now have
  `LLM_TIMEOUT_S` (15 s).
- **Verified:** the test fails on the old code (`checkin.active` stays True) and passes on the fixed code.

## 2 — Vision phone check dies after one stalled request (Medium, R1)
- **Repro:** `test_bug2_…`: the `VisionPhoneCheck` client had `http_options.timeout == None`.
- **Root cause:** `_busy` is cleared only when the request returns. A request that hangs keeps it
  set, so `maybe_check` returns early for the rest of the session.
- **Fix:** `phone_check.py`: `VISION_TIMEOUT_S = 15` on the client.
- **Verified:** the test fails before the fix (`None == 15000`) and passes after it.

## 3 — State push can die permanently (Low, R4 hardening)
- **Repro:** `test_bug3_…`: broadcasting a state that contains `np.float32` raised `TypeError`.
  The exception escaped the `_state_pump` task, so no more state would reach the dashboards.
- **Root cause:** only `pulse` goes through `_jsonable`, and the pump loop had no error handling.
  No current producer of a numpy value was found, so this is a latent bug, not an observed one.
- **Fix:** `server.py`: `json.dumps(default=_json_default)` converts numpy values to plain numbers and lists, and
  the pump logs a failed push and keeps running.
- **Verified:** the test fails before the fix and passes after it.

## 4 — Stale detections while the camera is disconnected (Low, R4)
- **Repro:** `test_bug4_…`: after `_publish_no_camera()` the state still held the last
  `objects` and `hands` values ("2 seen · 1 free", phone source "yolo").
- **Root cause:** `_publish_no_camera` reset attention and face fields but merged everything else from the last frame.
- **Fix:** `pipeline.py`: the no-camera state clears `objects` and the hand counts.
- **Verified:** the test fails before the fix and passes after it.

## 5 — Ping timers pile up on reconnect (Low, web)
- **Root cause:** `connect()` in `index.html` and `copilot.html` started a new `setInterval` on every
  reconnect and never cleared the old one.
- **Fix:** each socket clears its own timer when it closes.
- **Verified:** by reading the code (the frontend has no JS test harness).

## Final re-audit and what it did not cover
Re-read for this audit: `pipeline.py`, `server.py`, `config.py`, `__main__.py`, `camera.py`, `llm/coach.py`,
`llm/intervention.py`, `llm/dialogue.py`, `vision/attention.py`, `vision/driver.py`, `vision/objects.py`,
`vision/phone_check.py`, the key functions of `vision/hands.py`, and the WebSocket, speech and fetch code of both pages.
No further Critical, High or Medium issues were found.

**Not audited in depth (so no claim is made about them):** `vision/rppg.py`, `filters.py` and `identity.py` (ported,
validated code; `rppg` access is lock-guarded); `vision/face.py` beyond the bbox and contour types; `vision/discomfort.py`;
the experimental `RGB/` module (off by default); and most of the dashboard's rendering JS.
Nothing was tested against a live camera or the real Gemini API.

Remaining minor items (accepted, Low):
- While a check-in round is being judged (at most 6 s now), only a microsleep can interrupt. This is the intended design.
- `copilot.html` decides "old history versus new message" by comparing the server's clock with the browser's. This only matters if the page runs on another device.
