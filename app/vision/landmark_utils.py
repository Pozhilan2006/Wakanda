from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot
from typing import Iterable


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def distance(first: Landmark, second: Landmark) -> float:
    return hypot(first.x - second.x, first.y - second.y)


def dot(first: Landmark, second: Landmark) -> float:
    return first.x * second.x + first.y * second.y


def subtract(first: Landmark, second: Landmark) -> Landmark:
    return Landmark(first.x - second.x, first.y - second.y, first.z - second.z)


def angle_degrees(first: Landmark, vertex: Landmark, second: Landmark) -> float:
    """Return the angle at vertex in the normalized landmark plane."""
    first_vector = subtract(first, vertex)
    second_vector = subtract(second, vertex)
    denominator = distance(first, vertex) * distance(second, vertex)
    if denominator <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(first_vector, second_vector) / denominator))
    return degrees(acos(cosine))


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def midpoint(first: Landmark, second: Landmark) -> Landmark:
    return Landmark(
        (first.x + second.x) / 2.0,
        (first.y + second.y) / 2.0,
        (first.z + second.z) / 2.0,
        min(first.visibility, second.visibility),
    )


def normalize_landmark(landmark: object, width: int, height: int) -> Landmark:
    """Convert pixel coordinates to normalized coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return Landmark(
        float(getattr(landmark, "x")) / width,
        float(getattr(landmark, "y")) / height,
        float(getattr(landmark, "z", 0.0)),
        float(getattr(landmark, "visibility", 1.0)),
    )


def average_visibility(landmarks: Iterable[Landmark]) -> float:
    values = [landmark.visibility for landmark in landmarks]
    return sum(values) / len(values) if values else 0.0
