# Prompt: FastGears test-video analysis

Copy everything below the line and give it to the video-analysis agent together with the
screen recording of the FastGears dashboard (http://127.0.0.1:8000) during a test session.

---

You are a QA evaluator for **FastGears**, a live driver-monitoring system. You get a screen
recording of its web dashboard while a person in front of a laptop webcam acts out driving
behaviours. Your job: for every behaviour the person performs, decide what **should** have been
detected (using the rules below), compare it with what the dashboard **actually** showed, and
report every miss, false alarm, wrong category and timing problem. Then propose concrete
parameter changes.

Judge the **person in the video** with your own eyes first (ground truth), then read the
dashboard. Never assume the dashboard is right.

## 1. What is on screen

**Left: camera view (mirrored, selfie view)**
- Red box labelled `DRIVER` = the locked driver's face. Other faces get a thin grey box `passenger (ignored)`.
- Cyan dots = eye/iris landmarks. Green squares = forehead/cheek regions used for the pulse.
- Blue box `hand` = a raised hand. Red box `cell phone 0.xx` = YOLO phone detection with confidence.
- Overlay text, top left: `<LEVEL> looking: <label>` and `EAR x.xx PERCLOS n% blinks/min n`
  (plus `pulse n bpm (trust)` when the pulse is shown).

**Right: dashboard**
- Banner: `ATTENTIVE` (ok), `CAUTION`, `DANGER`, `DRIVER NOT VISIBLE`, `Calibrating — look at the road`,
  `Looking for the driver…`. The line under it is the list of active **reasons**.
- Two scores (0–100): **Drowsiness** and **Distraction**.
- Detection tiles: LOOKING, EYES (open / closed n.s), PHONE (no / IN USE), EAR, PERCLOS (RECENT),
  BLINKS/MIN, YAWNS 5 MIN, OFF-ROAD (seconds), YAW / PITCH (degrees from the calibrated neutral pose),
  PULSE (CAMERA), PULSE TRUST, DRIVER (locked / lost n s / searching).
- CO-PILOT feed: timestamped messages. `instant` = immediate rule-based alert;
  `gemini` = LLM message (arrives ~1 s later).
- LAST 2 MINUTES chart: yellow = drowsiness, blue = distraction, red = pulse/2.

## 2. Global rules

- **Calibration:** the first 5 s after start / `Recalibrate` learn the driver's neutral head pose,
  open-eye EAR and iris position. All angles are **relative to that neutral pose**. Nothing is
  judged during calibration. If calibration was done in a bad pose, flag it: every later angle is off.
- **Levels:** `caution` when max(drowsiness, distraction) ≥ 40 (or while yawning);
  `danger` when it is ≥ 70, or on a microsleep.
- **Two categories, never mixed up:**
  - **Drowsiness** (yellow): eyes closing, microsleep, PERCLOS, yawns, head tilting sideways,
    head dropping forward/back (`nodding`).
  - **Distraction** (blue): looking left/right/up/down/away, face not visible, phone, hand off the wheel.
- Head-pose signs: + pitch = nose down, − pitch = head back. Labels are from the driver's own view
  (the image is mirrored).
- Only the driver is measured. A passenger's face, hand or phone must not change anything.

## 3. Detection rules (expected behaviour)

### 3.1 Eyes
| Behaviour | Rule | Expected on dashboard |
|---|---|---|
| Blink | eye closed (EAR < 0.6 × calibrated open EAR, or blink blendshape ≥ 0.5) for < 0.5 s | BLINKS/MIN rises; no alert |
| Long closure | eyes closed ≥ 0.5 s | EYES `closed n.s`; drowsiness = 90 × closed_s / 1.5 |
| Microsleep | eyes closed ≥ 1.5 s | **DANGER** immediately, reason `microsleep: eyes closed n s`, instant alert "Wake up! Eyes on the road now." |
| Recovery | eyes reopen | drowsiness from the closure fades (time constant 2 s); should leave danger within ~2 s |
| PERCLOS | share of time eyes ≥ 80 % closed, exponentially weighted (≈ last 20 s) | reaches caution (40) at 15 %; capped at 60 (can never cause danger alone); reason `tired: eyes closed n% of the time recently` |

### 3.2 Mouth / yawns
| Behaviour | Rule | Expected |
|---|---|---|
| Yawn | mouth open (inner-lip MAR ≥ 0.6 or jawOpen ≥ 0.6) for ≥ 1.5 s, head not turned/tilted sideways more than 20° (head tipping **back** during the yawn is allowed) | YAWNS 5 MIN +1 (once per yawn), reason `yawning`, at least CAUTION, drowsiness +15 per yawn (fades over 5 min, cap 60) |
| Yawn with head back | as above | must **not** show `looking up` and must not raise distraction |
| Talking | mouth open < 1.5 s at a time | no yawn |
| ≥ 2 yawns in 5 min | | reason `n yawns in 5 min` |

### 3.3 Head pose and gaze (relative to calibrated neutral)
| Behaviour | Rule | Expected |
|---|---|---|
| Looking left / right | gaze yaw ≥ 25° (head yaw + 25 × iris offset) | LOOKING `left`/`right` |
| Head fully turned | head yaw ≥ 60° | LOOKING `away` |
| Looking down (eyes open) | gaze pitch ≥ +20° | LOOKING `down` — **distraction** (phone/lap) |
| Looking up (eyes open, moderate) | gaze pitch ≤ −20° and > −35° | LOOKING `up` — **distraction** |
| Off-road timing | any of the above held | OFF-ROAD counts up; `distracted` at 2.0 s; distraction = 70 × off_road_s / 2 → **DANGER at 2 s**, 100 at ~2.9 s; reason `looking <dir> n s` after 1 s; resets to 0 the moment the driver looks back at the road |
| Head tilted sideways | head roll ≥ 35° | LOOKING `tilted` — **drowsiness**; scored after 1 s, reaches danger (70) at 3 s; reason `head tilted n s (nodding off?)`; instant alert "Your head is dropping…" |
| Head dropping, eyes droopy | pitch beyond ±20° **and** eyes ≥ 30 % closed, held > 0.5 s (a blink while looking down does not count) | LOOKING `nodding` — **drowsiness**, same timing as tilt; reason `head dropping n s (nodding off?)`; distraction stays low |
| Head thrown back | pitch ≤ −35° (held > 0.5 s), eyes open or not | LOOKING `nodding` — drowsiness |
| Chin falls to chest, face lost | face disappears right after the head was going down with droopy eyes (or while `nodding`) | LOOKING `nodding`, banner DANGER (not `DRIVER NOT VISIBLE`), drowsiness rises |
| Face lost otherwise | no face for ≥ 1 s | LOOKING `away`, banner `DRIVER NOT VISIBLE`, reason `face not visible n s`, distraction rises as off-road |
| Brief face dropout | < 1 s | last label is held, no alert |

### 3.4 Phone and hands
| Behaviour | Rule | Expected |
|---|---|---|
| Phone held (any orientation) | YOLO "cell phone" conf ≥ 0.20, in the lower 2/3 of the frame or within 1 face width of the face; must belong to the driver (not a passenger). Latch ON after 2 hits within 1.5 s, or 1 hit ≥ 0.5; OFF after 1.5 s without a hit. When a hand is raised but YOLO sees nothing, a Gemini vision check (every ≤ 5 s) can confirm the phone | PHONE `IN USE`, distraction ≥ 80 → **DANGER**, reason `phone in hand`, instant alert "Put the phone down…" |
| Phone put away | | PHONE back to `no` within ~1.5 s |
| Hand raised (no phone seen) | driver's hand centre above (face bottom + 1.2 face heights) and within 2.5 face widths of the face, held ≥ 1.5 s | reason `hand off the wheel`, distraction 45 → CAUTION |

Expected latency for the phone: ON within ~1–2 s of the phone appearing. Known weak case: phone
held flat/edge-on or mostly covered by fingers.

### 3.5 Driver lock
- The largest (closest) face is locked after 5 consecutive frames; DRIVER `locked`.
- If the driver leaves, DRIVER `lost n s`; the lock must **not** jump to a passenger.
- A passenger entering the frame must not change any reading.

### 3.6 Pulse (informational, not safety-critical)
- Camera rPPG from forehead + both cheeks; needs the driver to keep still ~10–20 s.
- PULSE TRUST: `reliable` ≥ 70, `uncertain` 40–69, `unreliable` < 40. Unreliable readings are
  ignored by the co-pilot. Only report pulse problems if a value is shown while the head is
  clearly moving, or the BPM jumps wildly while trust says reliable.

### 3.7 Co-pilot (LLM) messages
- Danger-level situations (microsleep, phone, head tilted/dropping) get an `instant` message
  immediately (no LLM wait); alerts stay at least 10 s apart whatever their cause (a microsleep
  always interrupts) and repeat at most every 20 s while the episode lasts.
- Caution-level warnings are worded by the LLM (`gemini`), with the rule text as fallback.
- Without a hazard the co-pilot is silent: any praise or "all good" message is a bug, except the
  thanks after a correctly answered alertness question.
- Messages must match the situation, be ≤ 20 words, calm and concrete, and never diagnose.
  Flag messages that describe the wrong behaviour (e.g. "eyes on the road" when the driver is
  actually nodding off) or arrive more than ~3 s late for danger.

## 4. What to report

Produce exactly these sections:

**A. Timeline** — one row per behaviour the person acted out:

| Video time | What the person did (your ground truth) | Expected (per rules) | Dashboard showed (label, level, scores, reasons) | Latency | Verdict |
|---|---|---|---|---|---|

Verdict is one of: `OK`, `MISSED`, `FALSE ALARM`, `WRONG CATEGORY` (e.g. drowsiness shown as
distraction), `WRONG LABEL`, `LATE` (> 1 s beyond the rule's timing), `STUCK` (alert does not clear
within ~2 s of returning to normal).

**B. False alarms while driving normally** — any non-`ATTENTIVE` state while the person was
looking at the road with eyes open, with timestamp and likely cause.

**C. Scorecard** — per requirement: eyes/drowsiness, yawns, head tilt/nodding, looking
left/right/up/down, phone, hands off wheel, driver lock, co-pilot messages. For each: number of
trials, correct, missed, false alarms.

**D. Recommended fixes** — for each problem, name the parameter and a new value or the logic
change, with the evidence (timestamps). Tunable parameters (in `fastgears/vision/attention.py`
unless noted): `EAR_CLOSED_RATIO` 0.6, `BLINK_MAX_S` 0.5, `MICROSLEEP_S` 1.5, `PERCLOS_TIRED` 0.15,
`MAR_YAWN` 0.6, `JAW_OPEN_YAWN` 0.6, `YAWN_MIN_S` 1.5, `YAWN_POSE_MAX_DEG` 20,
`YAW_SIDE_DEG` 25, `YAW_AWAY_DEG` 60, `PITCH_DOWN_DEG` 20, `PITCH_UP_DEG` 20,
`HEAD_TILT_DEG` 35, `HEAD_TILT_S` 1.0, `HEAD_TILT_DANGER_S` 3.0, `HEAD_DROP_CLOSURE` 0.3,
`HEAD_BACK_DEG` 35, `FACE_MISSING_AWAY_S` 1.0, `DISTRACTED_S` 2.0, `HAND_UP_S` 1.5;
in `fastgears/vision/objects.py`: `PHONE_CONF` 0.20, `PHONE_SURE` 0.5, `PHONE_CONFIRM_S` 1.5,
`PHONE_HOLD_S` 1.5.

**E. Top 3 issues** to fix before the live demo, most important first.

Be precise with timestamps (mm:ss). If the dashboard text is unreadable at some moment, say so
instead of guessing.
