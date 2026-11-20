from types import SimpleNamespace

from fastgears.vision.driver import DriverLock

SHAPE = (720, 1280, 3)


def face(cx, w, cy=360):
    return SimpleNamespace(bbox=(cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2))


def feed(lock, frames, t0=0.0, dt=0.05):
    out, t = None, t0
    for faces in frames:
        t += dt
        out = lock.update(faces, SHAPE, t)
    return out, t


def test_locks_on_biggest_face_after_a_few_frames():
    lock = DriverLock()
    drv, pas = face(640, 220), face(200, 130)
    assert lock.update([pas, drv], SHAPE, 0.0) is None          # not locked on the first frame
    out, _ = feed(lock, [[pas, drv]] * 5)
    assert out is drv and lock.state.status == "locked" and lock.state.passengers == 1


def test_order_shuffle_and_small_moves_keep_the_driver():
    lock = DriverLock()
    feed(lock, [[face(200, 130), face(640, 220)]] * 6)
    for i in range(40):
        d, p = face(640 + 3 * i, 220), face(200 - i, 130)
        out = lock.update([d, p] if i % 2 else [p, d], SHAPE, 1 + i * 0.05)
        assert out is d


def test_never_jumps_to_passenger_when_driver_leaves():
    lock = DriverLock()
    feed(lock, [[face(200, 130), face(640, 220)]] * 6)
    out, t = feed(lock, [[face(200, 130)]] * 600, t0=1.0)       # 30 s with only the passenger
    assert out is None and lock.state.status == "lost" and lock.state.lost_s > 25


def test_passenger_leaning_into_view_is_not_taken_while_driver_away():
    lock = DriverLock()
    feed(lock, [[face(200, 130), face(640, 220)]] * 6)
    out, _ = feed(lock, [[face(420, 160)]] * 20, t0=1.0)        # someone between seats
    assert out is None


def test_driver_back_in_seat_is_reacquired():
    lock = DriverLock()
    feed(lock, [[face(200, 130), face(640, 220)]] * 6)
    feed(lock, [[face(200, 130)]] * 40, t0=1.0)
    back = face(660, 210)
    out, _ = feed(lock, [[face(200, 130), back]], t0=5.0)
    assert out is back and lock.state.status == "locked"


def test_seat_setting_overrides_size():
    lock = DriverLock(seat="left")
    small_left = face(300, 150)
    out, _ = feed(lock, [[face(900, 260), small_left]] * 6)
    assert out is small_left


def test_phone_ownership():
    lock = DriverLock()
    drv = face(640, 220)
    feed(lock, [[face(200, 130), drv]] * 6)
    assert lock.belongs_to_driver(700, 600, drv, SHAPE)          # in front of the driver
    assert not lock.belongs_to_driver(220, 600, drv, SHAPE)      # in the passenger's hand
    assert not lock.belongs_to_driver(700, 600, None, SHAPE)


def test_reset_reacquires():
    lock = DriverLock()
    feed(lock, [[face(640, 220)]] * 6)
    lock.reset()
    assert lock.state.status == "searching"
    out, _ = feed(lock, [[face(300, 240)]] * 5)
    assert out is not None and lock.state.status == "locked"


def test_driver_hand_next_to_passenger_face_is_still_the_drivers():
    # screenshot: driver (centre) raises a hand right beside the passenger's face (left)
    from fastgears.vision.hands import Hand
    lock = DriverLock()
    drv, pas = face(540, 140, cy=585), face(140, 150, cy=640)
    feed(lock, [[pas, drv]] * 6)
    # fingers up, wrist at the bottom: the forearm points down towards the driver's shoulder
    hand = Hand((200, 330, 410, 560), 0.9, wrist=(330, 560), forearm=(0.24, 0.97), length=120)
    assert lock.owns_hand(hand, drv)


def test_passenger_hand_stays_the_passengers():
    from fastgears.vision.hands import Hand
    lock = DriverLock()
    drv, pas = face(540, 140, cy=585), face(140, 150, cy=640)
    feed(lock, [[pas, drv]] * 6)
    hand = Hand((60, 380, 200, 560), 0.9, wrist=(130, 560), forearm=(0.0, 1.0), length=110)
    assert not lock.owns_hand(hand, drv)
