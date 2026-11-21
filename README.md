# AI_FastGears — AI Driver Monitoring Assistant

Codelap Hackathon 2026 · **AI Drive** challenge · Team FastGears

A live co-pilot that **watches the driver through a webcam**, understands when they are drowsy,
distracted or on the phone, and **reacts inside the car with an LLM**: spoken alerts, warnings and a
conversation, all shown on a **live dashboard**. Runs on a normal laptop: no Jetson, no GPU, no extra
sensors.

| Requirement | How FastGears covers it |
|---|---|
| **R1 Driver detection** | MediaPipe face mesh (478 landmarks + iris): eyes closed, blink rate, **PERCLOS**, microsleep, yawns, head yaw/pitch/roll, gaze → *looking left / right / down / up / away*. YOLOv8n → **phone in hand**. Contact-free **heart rate from the camera** (rPPG, POS algorithm). |
| **R2 In-car reaction (LLM)** | Instant local alert for danger (zero latency, works offline), then **Gemini** writes a short contextual message (info / warning / alert) that is spoken by the browser's voice. Driver or passenger can **talk to the co-pilot** (text or microphone). |
| **R3 Live demo** | Everything runs live on the webcam feed. Nothing is pre-recorded. |
| **R4 Live dashboard** | Browser dashboard: annotated live video, attention level banner, drowsiness/distraction gauges, every detection, 2-minute timeline, co-pilot feed with voice. |
| Optional GenAI | Natural-language explanations of *why* the level changed; passenger Q&A grounded on the live state. |

## Architecture

```
 webcam ──► Camera thread (newest frame only)
              │
              ▼
           Pipeline thread ───────────────────────────────────────────────┐
              │ FaceTracker (MediaPipe FaceLandmarker, VIDEO mode)         │
              │   landmarks, iris, blendshapes, head pose                  │
              ├─► RPPG (POS, forehead + cheeks, 10 s window) ─► pulse+trust │
              ├─► ObjectDetector (YOLOv8n every N frames) ─► phone/person  │
              └─► AttentionMonitor (calibrated to this driver)             │
                     EAR, PERCLOS, blinks, yawns, looking, off-road time   │
                     ─► level ok / caution / danger + reasons              │
                                   │                                       │
                                   ▼                                       ▼
                     Coach (LLM worker thread)                annotated JPEG
                     instant alert ─┐  Gemini / Anthropic / rules
                                    ▼
          FastAPI ── WebSocket /ws (state 5 Hz + messages) ── MJPEG /video.mjpg
                                    ▼
          Dashboard (browser): video, gauges, detections, timeline, co-pilot + voice
```

| Path | What it is |
|---|---|
| `fastgears/__main__.py` | Entry point (`python -m fastgears`), CLI flags |
| `fastgears/config.py` | All settings from `.env` / environment |
| `fastgears/camera.py` | Threaded webcam capture (DirectShow on Windows) |
| `fastgears/pipeline.py` | Per-frame loop, overlay drawing, state assembly |
| `fastgears/vision/face.py` | MediaPipe face landmarks, EAR, MAR, iris, head pose |
| `fastgears/vision/attention.py` | Drowsiness / distraction logic → level + reasons |
| `fastgears/vision/objects.py` | YOLOv8n phone / person detection |
| `fastgears/vision/rppg.py` | Camera heart rate (POS) with trust score |
| `fastgears/vision/identity.py` | Optional driver recognition (InsightFace) |
| `fastgears/vision/filters.py` | One-Euro smoothing filter |
| `fastgears/llm/coach.py` | LLM co-pilot: alerts, summaries, chat |
| `fastgears/server.py` | FastAPI: dashboard, WebSocket, MJPEG, REST |
| `fastgears/web/index.html` | The dashboard (no build step, no CDN) |
| `tests/` | Unit tests (no camera or models needed) |

### Reused, already-validated components

The camera pulse, face pose and identity code is ported from our earlier project
(afferent, Junction 2026), where it ran live:

- `vision/rppg.py`: POS rPPG (Wang et al. 2017) with skin filtering, 3-region agreement and a trust
  score. Checked against a finger pulse sensor: **mean absolute difference 1.8 BPM, max 4 BPM**.
  The face is now located with MediaPipe instead of YOLO-pose, and it uses a single webcam.
- `vision/face.py`: MediaPipe landmarker wrapper and head-pose Euler extraction (verified).
- `vision/identity.py`, `vision/filters.py`: face-embedding identity lock and One-Euro filter.

## How attention is measured

- **Calibration (first 5 s):** the driver looks at the road. This sets their own open-eye EAR and
  neutral head pose, so glasses, face shape and camera position don't matter. The *Recalibrate*
  button repeats it.
- **Eyes closed:** EAR below 60 % of the driver's baseline, or the MediaPipe eyeBlink blendshape ≥ 0.5.
- **Microsleep:** eyes closed ≥ 1.5 s → **danger**.
- **PERCLOS:** the share of the last 60 s with the eyes ≥ 80 % closed. This is the standard
  drowsiness measure; > 15 % means tired.
- **Yawns:** mouth aspect ratio / jawOpen held ≥ 1 s, counted over 5 min.
- **Looking:** head yaw/pitch relative to the baseline, plus iris position → road / left / right /
  down / up / away. **Distracted** = off the road ≥ 2 s.
- **Phone:** YOLO "cell phone" box in the frame near the driver → **danger**.
- **Pulse:** shown only with its trust label. The LLM ignores "unreliable" readings.

Thresholds are constants at the top of `fastgears/vision/attention.py`.

## Setup

Requirements: **Python 3.10–3.12** (MediaPipe has no wheels for 3.13 yet), a webcam,
Chrome or Edge (for voice), and a Gemini API key (free tier works).

### Windows (PowerShell)

Pentru iPhone conectat prin USB pe Windows, vezi [ghidul de instalare și configurare Camo](README_CAMO.md).
Pentru observațiile experimentale de culoare facială, vezi [ghidul modulului RGB](RGB/README.md).

Panoul **Diagnosticele** afișează simultan pulsul estimat, semnele de somnolență, grimasa și schimbarea de culoare. Pornește cu `python -m fastgears --rgb` pentru a include culoarea și calibrează camera cu fața relaxată, în lumină stabilă.

Panoul **Alerte șofer** urmărește separat oboseala și privirea, mâinile ridicate și telefonul în mână. FastGears fixează șoferul după cinci cadre și folosește fața acestuia pentru măsurători. Poți fixa partea în care se află prin `DRIVER_SEAT=left` sau `right` față de imaginea afișată. `ENABLE_HANDS=0` dezactivează detectorul de mâini. YOLO caută telefonul și în decupaje locale ale mâinilor. Pentru verificarea opțională cu Gemini Vision, `PHONE_VISION_CHECK=1` și o cheie Gemini permit trimiterea unui decupaj al mâinii către Gemini când YOLO nu confirmă telefonul.

Pe Windows, poți selecta camera și după numele DirectShow cu `--camera "dshow:Camo"` dacă FFmpeg este disponibil. Configurația Camo prin index și `--camera-backend msmf` rămâne disponibilă.

Tiparul de posibil preleșin cere reducerea activă a colorației obrajilor, o închidere a ochilor de cel puțin 0,5 s și un puls estimat sub 50 bpm cu semnal fiabil. Pragurile sunt euristici de demo. Detectorul de puls urmărește 45–180 bpm și folosește o fereastră de aproximativ 10 s, deci poate întârzia sau rata o scădere bruscă. Panoul oferă semnale orientative; validarea clinică a regulii rămâne deschisă.

```powershell
git clone --branch mara-test-rgb https://github.com/bbeatricecretu/AI_FastGears.git
cd AI_FastGears
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
copy .env.example .env
notepad .env        # paste GEMINI_API_KEY=...
```

### macOS / Linux

```bash
git clone --branch mara-test-rgb https://github.com/bbeatricecretu/AI_FastGears.git && cd AI_FastGears
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env   # then set GEMINI_API_KEY
```

The `--extra-index-url` line installs the small CPU build of PyTorch, which YOLO needs. On first
run the models download into `models/` automatically: `face_landmarker.task` (~4 MB) and
`yolov8n.pt` (~6 MB).

**Gemini key:** create one at <https://aistudio.google.com/apikey> and put it in `.env` as
`GEMINI_API_KEY`. To use Claude instead: `pip install anthropic`, then set `LLM_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY`. With no key the app still runs, using built-in rule-based messages.

## Run

```powershell
.venv\Scripts\python -m fastgears
```

Open **http://127.0.0.1:8000** and click anywhere once, because browsers only allow speech after a
click. Then look at the road (the camera) for 5 s to calibrate.

Useful flags:

```powershell
.venv\Scripts\python -m fastgears --camera 1          # second webcam
.venv\Scripts\python -m fastgears --camera 1 --camera-backend msmf  # Camo on this test laptop
.venv\Scripts\python -m fastgears --no-objects        # skip YOLO (faster on weak laptops)
.venv\Scripts\python -m fastgears --no-rppg           # skip camera pulse
.venv\Scripts\python -m fastgears --no-llm            # rule-based messages only
.venv\Scripts\python -m fastgears --host 0.0.0.0      # show the dashboard on another device in the LAN
```

API: `GET /api/state` (latest detections as JSON) · `GET /api/messages` · `POST /api/chat {"text": "..."}` ·
`POST /api/recalibrate` · `GET /api/health` · `WS /ws` · `GET /video.mjpg`.

## Puls și regiuni faciale în testul Camo

Pentru Camo pe laptopul de test, pornește `fastgears` cu `--camera 1 --camera-backend msmf` și lasă estimarea pulsului activată. Pe flux, verde marchează fruntea și ambii obraji, albastru marchează nasul, iar roz marchează contururile exterior și interior ale gurii. `face_regions` din `/api/state` oferă punctele în coordonatele cadrului original. Zona buzelor se obține scăzând conturul interior din cel exterior. O regiune ieșită din cadru apare cu valoarea `null`.

Estimatorul folosește timpul monoton la care OpenCV a livrat cadrul, inclusiv latența camerei virtuale. Fereastra are 6–10 s. La configurația implicită de 20 Hz, sub 8 probe/s sau la un gol de peste 350 ms nu se afișează BPM. Consensul regional necesită cel puțin 12 probe/s și două regiuni cu cel puțin 85% probe valide și vârfuri POS la cel mult 6 BPM distanță. Eliminăm variația de culoare explicată de mișcarea și schimbarea scalei feței și acceptăm vârfuri secundare comune dacă au suficientă putere în fiecare regiune. Un consens bazat pe un vârf secundar are încredere cel mult `uncertain`. Între 8 și 12 probe/s sau dacă lipsește consensul, POS pe culorile agregate furnizează BPM când există semnal utilizabil. Această citire are încredere `unreliable`. Pragurile sunt tehnice și experimentale.

În regim cu încredere scăzută, afișajul folosește mediana a până la trei estimări consecutive și păstrează ultima estimare brută în `pulse.raw_bpm`. Pentru a verifica acuratețea, compară estimatorii pe aceeași secvență de cadre cu o serie BPM de referință înregistrată simultan. Raportează eroarea absolută medie pentru citirile disponibile și proporția de ferestre în care fiecare estimator a publicat BPM. `pulse.reason`, `pulse.fps`, `pulse.max_gap_s` și `pulse.region_quality` explică rezultatul fiecărei ferestre. Testele din `tests/test_rppg.py` includ semnale sintetice cu probe rare, regiuni divergente și o frunte supraexpusă.

## Verificarea gesturilor faciale

```powershell
.venv\Scripts\python -m fastgears --no-objects --no-rppg --no-llm
```

Deschide [dashboard-ul local](http://127.0.0.1:8000) și urmărește panoul **Gesturi faciale**. Stai la birou cu fața la cameră, ochii deschiși și expresia relaxată. Panoul afișează progresul până la acumularea a 5 s de cadre eligibile. Apasă **Recalibrate** la schimbarea persoanei sau a poziției camerei.

Încearcă mișcările separat, apoi coboară sprâncenele, strânge pleoapele și ridică buza superioară simultan. Poți încreți nasul în loc să ridici buza. Dacă semnalele trec pragurile timp de cel puțin 300 ms (`DISCOMFORT_OVERLAP_S`), panoul afișează **Grimasă compatibilă cu disconfort**, regiunile și durata. Relaxează fața pentru a încheia episodul sau ieși din cadru pentru a-l întrerupe. Închiderea ochilor și deschiderea gurii oferă context. Un episod confirmat apare în `discomfort.event` din `/api/state`, iar ultimele opt episoade în `discomfort.recent_events`.

| Marker | Sursa MediaPipe | Valoare afișată |
|---|---|---|
| AU4 · Sprâncene | `browDownLeft/Right` | Creșterea scorului mediu de coborâre față de repaus |
| AU6 · Obraji | `cheekSquintLeft/Right` și deplasarea verticală a pomeților | Semnalul cel mai pronunțat față de propriul prag. Deplasarea se raportează la distanța dintre ochi |
| AU7 · Pleoape | `eyeSquintLeft/Right` | Creșterea scorului mediu de strângere activă |
| AU9 · Nas | `noseSneerLeft/Right` și ridicarea aripilor nazale față de puntea nazală | Semnalul cel mai pronunțat față de propriul prag. Ridicarea se măsoară ca procent din distanța dintre ochi și se folosește ca aproximare a mișcării asociate încrețirii |
| AU10 · Buza superioară | `mouthUpperUpLeft/Right` | Creșterea scorului mediu de ridicare a buzei |
| AU43 · Pleoape | EAR stânga și dreapta din `FaceResult` | Cea mai mică scădere relativă dintre cele două deschideri, activă de la 80% |
| AU25–27 · Gură | MAR din `FaceResult` | Creșterea raportului deschidere/lățime față de repaus |

Scorurile sunt aproximări ale mișcării. Fiecare scor are un prag de demo din `vision/discomfort.py`, ridicat dacă variația la repaus este mare. EAR și MAR folosesc distanțe normalizate, invariante la rotația în plan. Micșorarea EAR este raportată la ochii deschiși, fără un reper calibrat de ochi complet închiși. MediaPipe furnizează câte un `FaceResult` pe cadru, pe imaginea oglindită dacă `MIRROR=1`. Pentru coeficienții pereche folosim media, independent de oglindire. [Denumirile oficiale ale coeficienților MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc)

Panoul semnalează punctele regionale în afara imaginii, capul prea rotit, fața prea mică și valorile lipsă. Scorurile nu oferă confirmare separată a vizibilității în spatele ochelarilor sau al mâinii. Un gol de peste 250 ms întrerupe episodul. Acoperirea episodului se referă la continuitatea dovezilor din cadrele procesate, la rezoluția fluxului camerei. După cel puțin 2 s fără față se cere un nou reper. Co-pilotul primește durata și markerii evenimentului, fără pixeli, și poate face o verificare neutră cu șoferul.

### Demo script (≈2 min, for the jury)

1. Sit normally: the banner says **ATTENTIVE**, the pulse appears after about 10 s.
2. Look left or at your lap for 3 s: **looking left/down → CAUTION**, and the voice tells you to look
   at the road.
3. Hold up a phone: **phone in use → DANGER**, an instant alert, then the LLM's message.
4. Close your eyes for 2 s: **microsleep → DANGER** ("Wake up!").
5. Yawn a few times or keep your eyes half-closed: PERCLOS rises and the co-pilot suggests a break.
6. Ask the co-pilot "How am I doing?" (mic or text). It answers from the live state.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

The tests run without a camera, models or network. They use a synthetic 72 BPM pulse video,
scripted attention scenarios, and a fake LLM for the coach.

## Getting good readings

- Put the camera at eye level in front of the driver, 0.5–1 m away, with the face fully visible.
- **Camera pulse:** needs a still face and **white** light. Coloured light, backlight or a
  flickering screen ruin it. SNR < 0 dB or trust "unreliable" means *don't trust it*. A light
  flashing at heart rate on the face fakes the result.
- Laptop CPU: 15–30 processed fps with YOLO every 3rd frame. If it is slow, use `--no-objects`
  or `YOLO_EVERY=5`.
- On Windows, if the camera won't open, close other apps that use it (Teams, Zoom, the browser).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No matching distribution found for mediapipe` | Use Python 3.10–3.12 |
| Dashboard shows "no camera" | Try `--camera 1`, and check camera privacy settings on Windows |
| LLM provider shows `rules` | `GEMINI_API_KEY` missing or invalid in `.env` |
| No voice | Click the page once and check the **Voice on** button; use Chrome or Edge |
| Phone never detected | Hold it visibly; good light helps; `YOLO_MODEL=yolov8s.pt` is more accurate but slower |
