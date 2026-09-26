from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.config.settings import GestureSettings
from app.vision.landmark_utils import Landmark, angle_degrees, average_visibility, clamp, distance, midpoint, subtract


LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass(frozen=True)
class PoseLandmarks:
    left_shoulder: Landmark
    right_shoulder: Landmark
    left_hip: Landmark = Landmark(0.0, 0.0, visibility=0.0)
    right_hip: Landmark = Landmark(0.0, 0.0, visibility=0.0)
    left_elbow: Landmark = Landmark(0.0, 0.0, visibility=0.0)
    right_elbow: Landmark = Landmark(0.0, 0.0, visibility=0.0)
    left_wrist: Landmark = Landmark(0.0, 0.0, visibility=0.0)
    right_wrist: Landmark = Landmark(0.0, 0.0, visibility=0.0)

    def values(self) -> tuple[Landmark, ...]:
        return tuple(getattr(self, name) for name in LANDMARK_NAMES)

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "x": getattr(self, name).x,
                "y": getattr(self, name).y,
                "z": getattr(self, name).z,
                "visibility": getattr(self, name).visibility,
            }
            for name in LANDMARK_NAMES
        }


@dataclass(frozen=True)
class GestureFeatures:
    shoulder_width: float
    torso_width: float
    torso_height: float
    left_wrist_torso_x: float
    right_wrist_torso_x: float
    left_wrist_torso_y: float
    right_wrist_torso_y: float
    left_wrist_chest_distance: float
    right_wrist_chest_distance: float
    left_wrist_horizontal: float
    right_wrist_horizontal: float
    left_wrist_vertical: float
    right_wrist_vertical: float
    left_elbow_angle: float
    right_elbow_angle: float
    left_arm_crossing: float
    right_arm_crossing: float
    wrist_crossing: float
    chest_position: float
    elbow_geometry: float
    arm_symmetry: float
    landmark_visibility: float
    arm_intersection: float
    pose_consistency: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class GestureResult:
    detected: bool
    confidence: float
    features: GestureFeatures | None = None


class GestureRecognizer(Protocol):
    def evaluate(self, pose: PoseLandmarks | None) -> GestureResult:
        ...


def _segment_distance(first: Landmark, second: Landmark, third: Landmark, fourth: Landmark) -> float:
    def point_on_segment(start: Landmark, end: Landmark, point: Landmark) -> Landmark:
        vector = subtract(end, start)
        segment_length_sq = vector.x * vector.x + vector.y * vector.y
        if segment_length_sq <= 1e-9:
            return start
        t = clamp(((point.x - start.x) * vector.x + (point.y - start.y) * vector.y) / segment_length_sq)
        return Landmark(start.x + t * vector.x, start.y + t * vector.y)

    candidates = [
        distance(first, third),
        distance(first, fourth),
        distance(second, third),
        distance(second, fourth),
    ]
    for start, end in ((first, second), (third, fourth)):
        for point in (third, fourth):
            candidate = point_on_segment(start, end, point)
            candidates.append(distance(point, candidate))
    return min(candidates)


class CrossedArmDetector:
    """Normalized rule recognizer; the detector is intentionally conservative and uses torso-relative geometry."""

    def __init__(self, settings: GestureSettings):
        self.settings = settings

    def extract_features(self, pose: PoseLandmarks) -> GestureFeatures:
        shoulder_mid = midpoint(pose.left_shoulder, pose.right_shoulder)
        left_hip = pose.left_hip if pose.left_hip.visibility > 0.0 else shoulder_mid
        right_hip = pose.right_hip if pose.right_hip.visibility > 0.0 else shoulder_mid
        hip_mid = midpoint(left_hip, right_hip)
        torso_center = midpoint(shoulder_mid, hip_mid)
        shoulder_width = max(distance(pose.left_shoulder, pose.right_shoulder), 1e-6)
        torso_width = max(shoulder_width, distance(left_hip, right_hip), 0.2)
        torso_height = max(distance(shoulder_mid, hip_mid), 0.2)

        torso_axis = subtract(pose.right_shoulder, pose.left_shoulder)
        torso_axis_length = max(distance(pose.left_shoulder, pose.right_shoulder), 1e-6)
        torso_axis = Landmark(torso_axis.x / torso_axis_length, torso_axis.y / torso_axis_length)
        torso_perp = Landmark(-torso_axis.y, torso_axis.x)

        def torso_local(point: Landmark) -> tuple[float, float]:
            relative = subtract(point, torso_center)
            return (
                relative.x * torso_axis.x + relative.y * torso_axis.y,
                relative.x * torso_perp.x + relative.y * torso_perp.y,
            )

        left_torso_x, left_torso_y = torso_local(pose.left_wrist)
        right_torso_x, right_torso_y = torso_local(pose.right_wrist)
        left_elbow_x, left_elbow_y = torso_local(pose.left_elbow)
        right_elbow_x, right_elbow_y = torso_local(pose.right_elbow)

        left_wrist_horizontal = left_torso_x / max(torso_width, 1e-6)
        right_wrist_horizontal = right_torso_x / max(torso_width, 1e-6)
        left_wrist_vertical = left_torso_y / max(torso_height, 1e-6)
        right_wrist_vertical = right_torso_y / max(torso_height, 1e-6)

        left_wrist_chest_distance = distance(pose.left_wrist, torso_center) / max(shoulder_width, 1e-6)
        right_wrist_chest_distance = distance(pose.right_wrist, torso_center) / max(shoulder_width, 1e-6)

        left_arm_crossing = clamp((left_torso_x + 0.12 * torso_width) / (0.42 * torso_width))
        right_arm_crossing = clamp((0.12 * torso_width - right_torso_x) / (0.42 * torso_width))
        wrist_crossing = (left_arm_crossing + right_arm_crossing) / 2.0

        chest_extent_x = 0.85 * torso_width
        chest_extent_y = 1.10 * torso_height
        left_chest_occupancy = (
            clamp(1.0 - abs(left_torso_x) / max(chest_extent_x, 1e-6))
            * clamp(1.0 - abs(left_torso_y) / max(chest_extent_y, 1e-6))
        )
        right_chest_occupancy = (
            clamp(1.0 - abs(right_torso_x) / max(chest_extent_x, 1e-6))
            * clamp(1.0 - abs(right_torso_y) / max(chest_extent_y, 1e-6))
        )
        chest_position = (left_chest_occupancy + right_chest_occupancy) / 2.0

        left_angle = angle_degrees(pose.left_shoulder, pose.left_elbow, pose.left_wrist)
        right_angle = angle_degrees(pose.right_shoulder, pose.right_elbow, pose.right_wrist)

        def elbow_support(angle: float) -> float:
            return clamp(1.0 - abs(angle - 110.0) / 90.0)

        elbow_geometry = (elbow_support(left_angle) + elbow_support(right_angle)) / 2.0

        distance_symmetry = clamp(1.0 - abs(abs(left_torso_x) - abs(right_torso_x)) / max(torso_width, 1e-6))
        height_symmetry = clamp(1.0 - abs(left_torso_y - right_torso_y) / max(torso_height, 1e-6))
        arm_symmetry = (distance_symmetry + height_symmetry) / 2.0

        visibility = average_visibility(pose.values())
        cross_distance = (distance(pose.left_wrist, pose.right_elbow) + distance(pose.right_wrist, pose.left_elbow)) / 2.0
        arm_intersection = clamp(1.0 - (cross_distance / max(0.35 * shoulder_width + 0.12, 1e-6)))
        pose_consistency = (wrist_crossing + chest_position + arm_intersection + arm_symmetry) / 4.0

        return GestureFeatures(
            shoulder_width=shoulder_width,
            torso_width=torso_width,
            torso_height=torso_height,
            left_wrist_torso_x=left_torso_x,
            right_wrist_torso_x=right_torso_x,
            left_wrist_torso_y=left_torso_y,
            right_wrist_torso_y=right_torso_y,
            left_wrist_chest_distance=left_wrist_chest_distance,
            right_wrist_chest_distance=right_wrist_chest_distance,
            left_wrist_horizontal=left_wrist_horizontal,
            right_wrist_horizontal=right_wrist_horizontal,
            left_wrist_vertical=left_wrist_vertical,
            right_wrist_vertical=right_wrist_vertical,
            left_elbow_angle=left_angle,
            right_elbow_angle=right_angle,
            left_arm_crossing=left_arm_crossing,
            right_arm_crossing=right_arm_crossing,
            wrist_crossing=wrist_crossing,
            chest_position=chest_position,
            elbow_geometry=elbow_geometry,
            arm_symmetry=arm_symmetry,
            landmark_visibility=visibility,
            arm_intersection=arm_intersection,
            pose_consistency=pose_consistency,
        )

    def evaluate(self, pose: PoseLandmarks | None) -> GestureResult:
        if pose is None:
            return GestureResult(False, 0.0)
        features = self.extract_features(pose)
        weights = self.settings.weights
        total_weight = sum(weights.values()) or 1.0
        confidence = (
            weights.get("crossing", weights.get("wrist_crossing", 0.0)) * features.wrist_crossing
            + weights.get("chest_occupancy", weights.get("chest_position", 0.0)) * features.chest_position
            + weights.get("forearm_intersection", 0.0) * features.arm_intersection
            + weights.get("elbow_geometry", 0.0) * features.elbow_geometry
            + weights.get("arm_symmetry", 0.0) * features.arm_symmetry
            + weights.get("landmark_visibility", 0.0) * features.landmark_visibility
        ) / total_weight

        detected = (
            confidence >= self.settings.confidence_threshold
            and features.left_arm_crossing >= self.settings.wrist_crossing_min
            and features.right_arm_crossing >= self.settings.wrist_crossing_min
            and features.chest_position >= self.settings.chest_position_min
            and features.arm_intersection >= 0.10
            and features.landmark_visibility >= self.settings.visibility_threshold
            and min(landmark.visibility for landmark in pose.values()) >= self.settings.visibility_threshold
            and features.pose_consistency >= 0.45
        )
        return GestureResult(detected, clamp(confidence), features)


class GestureState(str, Enum):
    IDLE = "IDLE"
    GESTURE_CANDIDATE = "GESTURE_CANDIDATE"
    WAKANDA_ACTIVE = "WAKANDA_ACTIVE"
    COOLDOWN = "COOLDOWN"


class GestureStateMachine:
    def __init__(self, settings: GestureSettings):
        self.settings = settings
        self.state = GestureState.IDLE
        self._positive_frames = 0
        self._negative_frames = 0
        self._cooldown_until = 0.0

    @property
    def positive_frames(self) -> int:
        return self._positive_frames

    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self._cooldown_until - now)

    def update(self, result: GestureResult, now: float) -> GestureState:
        if self.state is GestureState.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = GestureState.IDLE
            else:
                return self.state
        if result.detected:
            self._positive_frames += 1
            self._negative_frames = 0
            if self.state is GestureState.IDLE:
                self.state = GestureState.GESTURE_CANDIDATE
            if self.state is GestureState.GESTURE_CANDIDATE and self._positive_frames >= self.settings.confirmation_frames:
                self.state = GestureState.WAKANDA_ACTIVE
            return self.state
        self._negative_frames += 1
        self._positive_frames = 0
        if self.state is GestureState.GESTURE_CANDIDATE and self._negative_frames >= self.settings.release_frames:
            self.state = GestureState.IDLE
        elif self.state is GestureState.WAKANDA_ACTIVE and self._negative_frames >= self.settings.release_frames:
            self.state = GestureState.COOLDOWN
            self._cooldown_until = now + self.settings.cooldown_seconds
        return self.state
