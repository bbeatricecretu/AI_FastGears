"""Correlate live camera observations for the dashboard using demo thresholds."""


def _fresh(timestamp, now, max_age):
    return timestamp is not None and 0 <= now - timestamp <= max_age


def assess_driver(t, attention, gestures, rgb, pulse):
    if not attention.get("face_present"):
        return {"finding": "face_unavailable"}

    eye_closed_s = attention.get("eye_closed_s", 0)
    prolonged_closure = eye_closed_s >= 0.5
    drowsy = prolonged_closure or attention.get("perclos", 0) >= 0.15
    grimace = (gestures.get("event") or {}).get("state") == "active"
    expression = "grimace" if grimace else "neutral" if gestures.get("observable") else "unavailable"

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
            "eye_closed_s": eye_closed_s, "expression": expression,
            "color": color_state, "lighting": lighting, "pulse": pulse_state, "bpm": bpm}
