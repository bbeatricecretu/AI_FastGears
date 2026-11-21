"""Correlate live camera observations (biometrics) for the dashboard using demo thresholds.

Eyes and mouth come from main's AttentionMonitor (calibrated per-eye closure, PERCLOS,
yawns); the discomfort monitor does not measure them a second time."""
from .attention import LONG_CLOSURE_S, PERCLOS_TIRED, YAWNS_TIRED

# yawns in the last 5 min that alone bring main's drowsiness score to caution
YAWNS_DROWSY = YAWNS_TIRED


def _fresh(timestamp, now, max_age):
    return timestamp is not None and 0 <= now - timestamp <= max_age


def assess_driver(t, attention, gestures, rgb, pulse):
    if not attention.get("face_present"):
        return {"finding": "face_unavailable"}

    # eyes (bilateral + per eye) and mouth, as measured by AttentionMonitor
    eye_closed_s = attention.get("eye_closed_s", 0)
    prolonged_closure = eye_closed_s >= LONG_CLOSURE_S or bool(attention.get("microsleep"))
    yawning = bool(attention.get("yawning"))
    yawns = attention.get("yawns_5min") or 0
    drowsy = (prolonged_closure or attention.get("perclos", 0) >= PERCLOS_TIRED
              or yawns >= YAWNS_DROWSY)
    # a yawn squints the eyes and lifts the upper lip: not a discomfort grimace
    grimace = (gestures.get("event") or {}).get("state") == "active" and not yawning
    expression = ("yawn" if yawning else "grimace" if grimace
                  else "neutral" if gestures.get("observable") else "unavailable")

    lighting = rgb.get("lighting")
    observations = rgb.get("observations", {})
    color_state = "disabled" if not rgb.get("enabled") else "unavailable"
    if lighting in ("stable", "estimated") and _fresh(rgb.get("sampled_monotonic_s"), t, 0.35):
        for kind, signal in (("cheek_color_reduction", "pallor"), ("lip_tone_shift", "lip_tone"),
                             ("cheek_redness", "redness")):
            if observations.get(kind, {}).get("state") == "active":
                color_state = signal
                break
        else:
            if observations.get("cheek_color_reduction", {}).get("state") == "idle":
                color_state = "no_change"

    bpm = pulse.get("bpm")
    trusted = ((pulse.get("trust") or {}).get("label") == "reliable"
               and pulse.get("pulse_state", "tracking") in ("tracking", "held")
               and _fresh(pulse.get("sampled_monotonic_s"), t, 2.0))
    bpm = bpm if trusted else None
    pulse_state = "unavailable"
    if bpm is not None:
        pulse_state = "slow" if bpm < 50 else "fast" if bpm > 100 else "typical"

    color_changed = color_state in ("pallor", "lip_tone", "redness")
    pulse_outside_range = pulse_state in ("slow", "fast")
    if color_state == "pallor" and prolonged_closure:
        finding = "possible_presyncope" if pulse_state == "slow" else "pallor_and_eye_closure"
    elif drowsy:
        finding = "drowsiness"
    elif grimace and (color_changed or pulse_outside_range):
        finding = "combined_discomfort_signals"
    elif color_changed and pulse_outside_range:
        finding = "color_and_pulse_change"
    elif "unavailable" in (expression, color_state, pulse_state) or color_state == "disabled":
        finding = "partial_signals"
    else:
        finding = "no_combined_pattern"

    return {"finding": finding, "stop_safely": drowsy, "drowsy": drowsy,
            "eye_closed_s": eye_closed_s,
            "eyes": {"closed": bool(attention.get("eyes_closed")),
                     "left_closed": attention.get("eye_left_closed"),
                     "right_closed": attention.get("eye_right_closed"),
                     "perclos": attention.get("perclos")},
            "mouth": {"yawning": yawning, "yawns_5min": yawns},
            "expression": expression,
            "color": color_state, "lighting": lighting, "pulse": pulse_state, "bpm": bpm}
