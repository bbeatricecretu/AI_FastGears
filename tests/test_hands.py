"""Hands on / off the wheel (no models: synthetic landmarks)."""
from types import SimpleNamespace

from fastgears.vision.hands import Hand, free_hands, hand_free, openness

SHAPE = (720, 1280, 3)
FACE = SimpleNamespace(bbox=(400, 150, 520, 280))       # driver face, chin at y=280


def _pts(open_hand: bool, x=300.0, y=600.0):
    """21 landmarks, fingers pointing up from the wrist at (x, y)."""
    xs, ys = [x] * 21, [y] * 21
    for f, (k, tip) in enumerate(zip((5, 9, 13, 17), (8, 12, 16, 20))):
        dx = (f - 1.5) * 12
        xs[k], ys[k] = x + dx, y - 50                     # knuckles 50 px above the wrist
        xs[tip], ys[tip] = x + dx, y - (100 if open_hand else 45)   # curled: tip back by the palm
    return xs, ys


def _hand(open_hand, x=300.0, y=600.0):
    xs, ys = _pts(open_hand, x, y)
    return Hand((int(min(xs)) - 10, int(min(ys)) - 10, int(max(xs)) + 10, int(y) + 5), 0.9,
                openness=openness(xs, ys))


def test_openness_separates_open_and_gripping():
    assert openness(*_pts(True)) > 1.8
    assert openness(*_pts(False)) < 1.1


def test_gripping_hand_is_on_the_wheel_open_hand_is_free():
    assert not hand_free(_hand(False), FACE, SHAPE)
    assert hand_free(_hand(True), FACE, SHAPE)


def test_hand_at_the_face_is_free_even_curled():
    assert hand_free(_hand(False, x=470, y=300), FACE, SHAPE)


def test_hand_cut_off_by_frame_edge_counts_as_on_wheel():
    h = _hand(True, y=715)
    h.bbox = (h.bbox[0], h.bbox[1], h.bbox[2], 719)
    assert not hand_free(h, FACE, SHAPE)


def test_one_hand_on_wheel_is_not_both_free():
    hands = [_hand(True, x=250), _hand(False, x=650)]
    assert len(free_hands(hands, FACE, SHAPE)) == 1
    assert len(free_hands([_hand(True, x=250), _hand(True, x=650)], FACE, SHAPE)) == 2
    assert free_hands(hands, None, SHAPE) == []


def test_phone_hit_on_open_palm_is_vetoed_but_not_on_gripping_hand():
    from fastgears.vision.hands import on_open_palm
    palm, grip = _hand(True, x=300), _hand(False, x=300)
    hit = (285, 520, 320, 570)                                   # on the hand
    assert on_open_palm(hit, [palm])
    assert not on_open_palm(hit, [grip])                         # a hand around a real phone
    assert not on_open_palm((900, 520, 940, 570), [palm])        # elsewhere
    warped = Hand(palm.bbox, 0.9, openness=2.8)                  # phone hides fingers: landmarks warp
    assert not on_open_palm(hit, [warped])


def test_phone_box_around_the_drivers_head_is_not_a_phone():
    from fastgears.vision.objects import ObjectDetector
    assert ObjectDetector.covers_face((380, 100, 540, 330), FACE)      # tall box over the head
    assert not ObjectDetector.covers_face((520, 160, 580, 280), FACE)  # phone at the ear, beside it
    assert not ObjectDetector.covers_face((380, 100, 540, 330), None)
