from types import SimpleNamespace

import pytest

from app.vision.landmark_utils import Landmark, average_visibility, distance, midpoint, normalize_landmark


def test_normalize_landmark_from_pixels():
    point = normalize_landmark(SimpleNamespace(x=640, y=360, z=-0.2, visibility=0.8), 1280, 720)
    assert point == Landmark(0.5, 0.5, -0.2, 0.8)


def test_normalize_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        normalize_landmark(SimpleNamespace(x=1, y=1), 0, 720)


def test_distance_midpoint_and_visibility():
    first = Landmark(0.0, 0.0, visibility=0.8)
    second = Landmark(3.0, 4.0, visibility=0.6)
    assert distance(first, second) == 5.0
    assert midpoint(first, second) == Landmark(1.5, 2.0, visibility=0.6)
    assert average_visibility((first, second)) == pytest.approx(0.7)
