"""Driver lock: of all faces in the frame, only the driver's is measured.

In a car the driver never changes seat while driving, so the lock is *seat-based*:

1. Acquire: pick the driver among the visible faces. DRIVER_SEAT=left|right restricts the
   choice to that half of the (displayed) image; "auto" takes the biggest face (closest to
   the camera), ties going to the most central one. The same face must win LOCK_FRAMES
   frames in a row before the lock is set, so a face passing by is never locked.
2. Track: each frame, the driver is the face that continues the locked one (centre moved
   less than TRACK_MAX_MOVE face widths, similar size). MediaPipe's face order changes
   between frames, so order is never trusted.
3. Lost: if no face continues the driver, the driver is *not visible*. The lock does NOT
   jump to a passenger. Only a face back in the driver's seat (centre within SEAT_TOL of
   the learned seat position, similar size) re-acquires it.
4. Recalibrate (dashboard button) forgets the lock and acquires again.

Driver crop (roi/merge): a full-frame detector misses a face that is small in a wide view
(camera between the front seats, ~1 m away). Each frame the pipeline also searches a crop
around the driver's last position (or the seat half while acquiring) and merges that face in.

Other faces are reported as passengers so the overlay can show them as ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LOCK_FRAMES = 5           # consecutive wins needed to lock
TRACK_MAX_MOVE = 0.8      # max centre jump between frames, in face widths
SIZE_RATIO_MAX = 1.8      # width can change at most this factor frame-to-frame / on re-acquire
SEAT_TOL = 0.6            # re-acquire: |centre x - seat x| <= this many driver face widths
SEAT_EMA = 0.02           # seat position follows the driver slowly while tracked
MIN_FACE_FRAC = 0.04      # ignore faces narrower than 4 % of the frame (far background)
ROI_TRACK = 3.0           # driver crop while tracked: this many face widths square
ROI_LOST = 4.5            # wider crop around the seat while the driver is lost
ROI_MIN_PX = 224
SIDE_REACH = 1.5          # left/right seat: a hand or phone further than this many driver face
                          # widths past the image midline (or past the face, if that is further)
                          # is on the passenger's side - not the driver's, even with the
                          # passenger's face unseen


def _cx(f) -> float:
    x0, _, x1, _ = f.bbox
    return (x0 + x1) / 2.0


def _cy(f) -> float:
    _, y0, _, y1 = f.bbox
    return (y0 + y1) / 2.0


def _w(f) -> float:
    x0, _, x1, _ = f.bbox
    return max(1.0, float(x1 - x0))


@dataclass
class DriverState:
    status: str = "searching"          # searching | locked | lost
    lost_s: float = 0.0
    passengers: int = 0
    seat_x: float | None = None        # 0..1 of frame width
    passenger_boxes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"status": self.status, "locked": self.status != "searching",
                "lost_s": round(self.lost_s, 1), "passengers": self.passengers,
                "seat_x": None if self.seat_x is None else round(self.seat_x, 3)}


class DriverLock:
    def __init__(self, seat: str = "auto", lock_frames: int = LOCK_FRAMES):
        seat = (seat or "auto").lower()
        if seat not in ("auto", "left", "right"):
            raise ValueError("DRIVER_SEAT must be auto, left or right")
        self.seat = seat
        self.lock_frames = lock_frames
        self.reset()

    def reset(self) -> None:
        self._last = None             # last driver face (bbox) while tracked
        self._seat_x: float | None = None
        self._seat_w: float | None = None
        self._seat_y: float | None = None   # px, last driver face centre height
        self._cand = None
        self._cand_n = 0
        self._lost_since: float | None = None
        self._frame_w: float | None = None
        self.state = DriverState()

    # -- selection --------------------------------------------------------------
    def _eligible(self, faces, W):
        out = [f for f in faces if _w(f) >= MIN_FACE_FRAC * W]
        if self.seat == "left":
            out = [f for f in out if _cx(f) < W / 2]
        elif self.seat == "right":
            out = [f for f in out if _cx(f) >= W / 2]
        return out

    @staticmethod
    def _best_initial(faces, W):
        # biggest face (closest to the camera); ties -> most central
        return max(faces, key=lambda f: (round(_w(f) / (0.05 * W)), -abs(_cx(f) - W / 2)))

    @staticmethod
    def _continues(prev, f) -> bool:
        pw = _w(prev)
        moved = ((_cx(f) - _cx(prev)) ** 2 + (_cy(f) - _cy(prev)) ** 2) ** 0.5 / pw
        ratio = max(_w(f), pw) / min(_w(f), pw)
        return moved <= TRACK_MAX_MOVE and ratio <= SIZE_RATIO_MAX

    def _in_seat(self, f, W) -> bool:
        if self._seat_x is None:
            return False
        ratio = max(_w(f), self._seat_w) / min(_w(f), self._seat_w)
        return abs(_cx(f) - self._seat_x * W) <= SEAT_TOL * self._seat_w and ratio <= SIZE_RATIO_MAX

    def update(self, faces: list, frame_shape, t: float):
        """Return the driver's face (or None) and update self.state."""
        W = float(frame_shape[1])
        self._frame_w = W
        faces = list(faces or [])
        driver = None

        if self._seat_x is None:                                 # ── acquiring
            elig = self._eligible(faces, W)
            if elig:
                best = self._best_initial(elig, W)
                if self._cand is not None and self._continues(self._cand, best):
                    self._cand_n += 1
                else:
                    self._cand_n = 1
                self._cand = best
                if self._cand_n >= self.lock_frames:
                    driver = best
                    self._seat_x, self._seat_w = _cx(best) / W, _w(best)
                    self._seat_y = _cy(best)
            else:
                self._cand, self._cand_n = None, 0
        else:                                                    # ── locked
            if self._last is not None:
                cont = [f for f in faces if self._continues(self._last, f)]
                if cont:
                    driver = min(cont, key=lambda f: abs(_cx(f) - _cx(self._last)))
            if driver is None:                                   # lost: only the seat re-acquires
                seat = [f for f in faces if self._in_seat(f, W)]
                if seat:
                    driver = min(seat, key=lambda f: abs(_cx(f) / W - self._seat_x))
            if driver is not None:
                self._seat_x += SEAT_EMA * (_cx(driver) / W - self._seat_x)
                self._seat_w += SEAT_EMA * (_w(driver) - self._seat_w)
                self._seat_y = _cy(driver)

        if driver is not None:
            self._last = driver
            self._lost_since = None
            status, lost_s = "locked", 0.0
        elif self._seat_x is not None:
            self._last = None
            if self._lost_since is None:
                self._lost_since = t
            status, lost_s = "lost", t - self._lost_since
        else:
            status, lost_s = "searching", 0.0

        others = [f for f in faces if f is not driver]
        self.state = DriverState(status=status, lost_s=lost_s, passengers=len(others),
                                 seat_x=self._seat_x, passenger_boxes=[list(f.bbox) for f in others])
        return driver

    # -- driver crop -------------------------------------------------------------------
    def roi(self, frame_shape):
        """Where to search for the driver's face this frame: (x0, y0, x1, y1) px, or None."""
        H, W = float(frame_shape[0]), float(frame_shape[1])
        if self._last is not None:                       # tracked: around the last face
            cx, cy, side = _cx(self._last), _cy(self._last), ROI_TRACK * _w(self._last)
        elif self._seat_x is not None:                   # lost: around the seat, wider
            cx, cy, side = self._seat_x * W, self._seat_y or H / 2, ROI_LOST * self._seat_w
        elif self.seat == "left":                        # acquiring: the driver's half
            return (0, 0, int(W * 0.55), int(H))
        elif self.seat == "right":
            return (int(W * 0.45), 0, int(W), int(H))
        else:
            return None
        half = max(side, ROI_MIN_PX) / 2.0
        x0, y0 = max(0.0, cx - half), max(0.0, cy - half)
        x1, y1 = min(W, cx + half), min(H, cy + half)
        return int(x0), int(y0), int(x1), int(y1)

    @staticmethod
    def merge(faces: list, roi_face) -> list:
        """Add the crop's face to the full-frame faces, replacing the same face found there."""
        faces = list(faces or [])
        if roi_face is None:
            return faces
        rx, ry, rw = _cx(roi_face), _cy(roi_face), _w(roi_face)
        rest = [f for f in faces
                if ((_cx(f) - rx) ** 2 + (_cy(f) - ry) ** 2) ** 0.5 > 0.5 * max(rw, _w(f))]
        return [roi_face] + rest

    # -- ownership of things in the frame ----------------------------------------------
    def _on_passenger_side(self, x: float, driver) -> bool:
        """Beyond the driver's reach towards the passenger seat (left/right seat mode only)."""
        W = self._frame_w
        if W is None or self.seat == "auto":
            return False
        reach = SIDE_REACH * _w(driver)
        if self.seat == "left":
            return x > max(W / 2.0, _cx(driver)) + reach
        return x < min(W / 2.0, _cx(driver)) - reach

    def owns_hand(self, hand, driver) -> bool:
        """The driver's hand if its forearm points at the driver rather than at a passenger."""
        if driver is None or self._on_passenger_side(hand.centre[0], driver):
            return False
        from .hands import owner_distance
        mine = owner_distance(hand, driver.bbox)
        return all(mine <= owner_distance(hand, b) for b in self.state.passenger_boxes)

    def belongs_to_driver(self, x: float, y: float, driver, frame_shape) -> bool:
        """Is a point (phone / hand centre) the driver's rather than a passenger's?"""
        if driver is None or self._on_passenger_side(x, driver):
            return False
        dw = _w(driver)
        d_drv = abs(x - _cx(driver)) / dw
        for b in self.state.passenger_boxes:
            pw = max(1.0, b[2] - b[0])
            if abs(x - (b[0] + b[2]) / 2.0) / pw < d_drv:
                return False                                     # closer to a passenger
        return True
