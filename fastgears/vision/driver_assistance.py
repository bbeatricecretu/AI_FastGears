"""Driver-only tracking and independent alerts alongside the biometric pipeline."""
from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

import cv2

from .cezara_attention import AttentionMonitor
from .driver import DriverLock
from .hands import HandTracker, free_hands, hand_crop, on_open_palm
from .objects import PhoneLatch

log = logging.getLogger(__name__)


class DriverAssistance:
    def __init__(self, settings, errors):
        self.errors = errors
        self.driver = DriverLock(settings.driver_seat)
        self.attention = AttentionMonitor(calib_s=settings.calib_s, mirrored=settings.mirror)
        self.hands = self.phone_vision = None
        self.objects_failed = False
        if settings.enable_hands:
            try:
                self.hands = HandTracker(models_dir=settings.models_dir, num_hands=4)
            except Exception as exc:
                self.errors["hands"] = str(exc)
                log.exception("hand tracker unavailable")
        if (settings.enable_objects and settings.phone_vision_check
                and settings.llm_provider == "gemini" and settings.gemini_api_key):
            try:
                from .phone_check import VisionPhoneCheck
                self.phone_vision = VisionPhoneCheck(settings.gemini_api_key, settings.gemini_vision_model)
            except Exception as exc:
                self.errors["phone_vision"] = str(exc)
                log.exception("vision phone check unavailable")
        self.reset()

    def reset(self):
        self.driver.reset()
        self.attention.reset_calibration()
        self.frame_count = 0
        self.last_scene_at = None
        self.passengers = []
        self.pitch_history = deque()
        self.posture_since = self.last_nod = self.posture = None
        self.clear_hands()

    def clear_hands(self):
        self.observed_hands = self.driver_hands = self.free = []
        self.hands_off_since = self.phone_xy = self.phone_source = None
        self.latch = PhoneLatch()
        if self.phone_vision:
            self.phone_vision.reset()

    def select_face(self, tracker, frame, t, t_ms):
        self.frame_count += 1
        faces = tracker.process_all(frame, t_ms) if tracker else []
        scan_scene = tracker and (self.last_scene_at is None or t - self.last_scene_at >= 0.6)
        if scan_scene:
            self.last_scene_at = t
            try:
                scene = tracker.process_scene(frame)
                faces = self.driver.merge(scene, faces[0]) if faces else scene
            except Exception:
                log.exception("passenger face pass failed")
        if tracker and self.frame_count % 3 == 0 and self.driver.needs_crop(faces, frame.shape):
            roi = self.driver.roi(frame.shape)
            if roi is not None:
                try:
                    faces = self.driver.merge(faces, tracker.process_roi(frame, roi))
                except Exception:
                    log.exception("driver crop face pass failed")
        driver = self.driver.update(faces, frame.shape, t)
        passenger_boxes = self.driver.state.passenger_boxes
        if scan_scene:
            self.passengers = [SimpleNamespace(bbox=box) for box in passenger_boxes]
        elif self.passengers:
            excluded = self.driver.merge(self.passengers, driver)[1:] if driver is not None else self.passengers
            for box in passenger_boxes:
                excluded = self.driver.merge(excluded, SimpleNamespace(bbox=box))
            self.driver.state.passenger_boxes = [list(passenger.bbox) for passenger in excluded]
            self.driver.state.passengers = len(excluded)
        return driver

    def update_objects(self, detector, frame, face, t, t_ms):
        if face is None:
            self.clear_hands()
        elif self.hands and self.frame_count % 3 == 0:
            try:
                self.observed_hands = self.hands.process(frame, t_ms)
            except Exception as exc:
                self.errors["hands"] = str(exc)
                log.exception("hand tracker stopped")
                self.hands = None
                self.observed_hands = []
        self.driver_hands = [hand for hand in self.observed_hands if self.driver.owns_hand(hand, face)]
        self.free = free_hands(self.driver_hands, face, frame.shape)
        if len(self.free) >= 2:
            if self.hands_off_since is None:
                self.hands_off_since = t
        else:
            self.hands_off_since = None
        boxes = [box for hand in self.driver_hands if (box := hand_crop(hand, frame.shape)) is not None]
        if detector is None or self.objects_failed:
            return [], False
        try:
            detections = detector.detect_live(frame, face, boxes, t)
        except Exception as exc:
            self.errors["objects"] = str(exc)
            self.objects_failed = True
            log.exception("object detector stopped")
            return [], False

        phones = []
        for item in detections:
            if item["label"] != "cell phone":
                continue
            x0, y0, x1, y1 = item["bbox"]
            center = (x0 + x1) / 2, (y0 + y1) / 2
            if (self.driver.belongs_to_driver(*center, face, frame.shape)
                    and not on_open_palm(item["bbox"], self.driver_hands)
                    and not detector.covers_face(item["bbox"], face)):
                phones.append(item)
        accepted = [item for item in detections if item["label"] != "cell phone"] + phones
        best = detector.phone_best(phones, face, frame.shape)
        hit = float(best["conf"]) if best else None
        fresh = detector.fresh
        if best and fresh:
            x0, y0, x1, y1 = best["bbox"]
            self.phone_xy = (x0 + x1) / 2, (y0 + y1) / 2
            self.phone_source = "yolo"
        if self.phone_vision and boxes and not self.latch.on and hit is None:
            self.phone_vision.maybe_check(frame, boxes[0], t)
            confidence = self.phone_vision.take_phone_conf(t)
            if confidence is not None:
                hit, fresh, self.phone_source = confidence, True, "gemini-vision"
                x0, y0, x1, y1 = boxes[0]
                self.phone_xy = (x0 + x1) / 2, (y0 + y1) / 2
        holding = self.phone_xy is not None and any(
            x0 <= self.phone_xy[0] <= x1 and y0 <= self.phone_xy[1] <= y1
            for x0, y0, x1, y1 in boxes)
        phone = face is not None and self.latch.update(hit, fresh, t, holding=holding)
        if not phone and hit is None:
            self.phone_xy = self.phone_source = None
        return accepted, phone

    def update_posture(self, face, attention, t):
        pitch, roll = attention.get("pitch"), attention.get("roll")
        label = None
        if face is not None and attention.get("calibrated"):
            base_ear = self.attention.base_ear
            ear = attention.get("ear")
            drooping = bool(base_ear and ear is not None and ear <= 0.79 * base_ear)
            if roll is not None and abs(roll) >= 35:
                label = "tilted"
            elif pitch is not None and (pitch <= -35 or abs(pitch) >= 20 and drooping):
                label = "nodding"
            if pitch is not None:
                self.pitch_history.append((t, pitch))
        while self.pitch_history and t - self.pitch_history[0][0] > 0.3:
            self.pitch_history.popleft()
        if label == "nodding":
            self.last_nod = t
        if face is None:
            recent_drop = (len(self.pitch_history) >= 2 and self.pitch_history[-1][1] >= 15
                           and self.pitch_history[-1][1] - self.pitch_history[0][1] >= 8)
            if recent_drop or self.last_nod is not None and t - self.last_nod <= 3:
                label = "nodding"
                if recent_drop:
                    self.last_nod = t
        if label != self.posture:
            self.posture, self.posture_since = label, t if label is not None else None
        duration = t - self.posture_since if self.posture_since is not None else 0.0
        return {"state": label, "duration_s": round(duration, 2), "active": duration >= 1,
                "severity": "alert" if duration >= 3 else "warning"}

    def state(self, face, attention, t, phone, objects_available):
        metrics = self.attention.update(face, t, phone=phone)
        posture = self.update_posture(face, attention, t)
        warning = metrics.get("active_warning")
        messages = [warning["message"]] if warning else []
        if posture["active"]:
            messages.append("Capul rămâne înclinat sau cade înainte.")
        glance_messages = {"sustained_off_road": "Privirea este menținută în afara drumului.",
                           "repeated_off_road": "Priviri repetate în afara drumului."}
        for event in metrics["events"]:
            if event["type"] in glance_messages and (not warning or event["id"] != warning.get("event_id")):
                messages.append(glance_messages[event["type"]])
        available = face is not None
        hands_off = self.hands_off_since is not None and t - self.hands_off_since >= 1.5
        return {
            "driver": self.driver.state.as_dict(),
            "hands": {"available": self.hands is not None, "seen": len(self.observed_hands),
                      "driver": len(self.driver_hands), "free": len(self.free),
                      "phone_source": self.phone_source if phone else None},
            "driver_attention": metrics,
            "driver_posture": posture,
            "driver_alerts": {
                "attention": {"active": True if messages else False if available and metrics["calibrated"] else None,
                              "messages": messages},
                "hands": {"active": hands_off if available and self.hands is not None else None},
                "phone": {"active": phone if available and objects_available else None},
            },
        }

    def draw(self, image):
        for x0, y0, x1, y1 in self.driver.state.passenger_boxes:
            cv2.rectangle(image, (x0, y0), (x1, y1), (145, 145, 145), 1)
            cv2.putText(image, "Passenger", (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (145, 145, 145), 1)
        free_ids = {id(hand) for hand in self.free}
        for hand in self.driver_hands:
            center = tuple(int(value) for value in hand.centre)
            color = (0, 190, 255) if id(hand) in free_ids else (90, 210, 110)
            cv2.circle(image, center, 9, color, 2)
