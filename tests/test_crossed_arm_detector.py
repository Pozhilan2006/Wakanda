import pytest

from app.config.settings import GestureSettings, PoseSettings
from app.vision.crossed_arm_detector import (
    CrossedArmDetector,
    GestureResult,
    GestureState,
    GestureStateMachine,
    PoseLandmarks,
)
from app.vision.landmark_utils import Landmark
from app.vision.pose_detector import PoseDetector


def crossed_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.43, 0.36),
        right_elbow=Landmark(0.57, 0.36),
        left_wrist=Landmark(0.59, 0.38),
        right_wrist=Landmark(0.41, 0.38),
    )


def normal_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.25, 0.45),
        right_elbow=Landmark(0.75, 0.45),
        left_wrist=Landmark(0.15, 0.60),
        right_wrist=Landmark(0.85, 0.60),
    )


def test_crossed_arm_confidence_and_rejection():
    settings = GestureSettings(confidence_threshold=0.55)
    result = CrossedArmDetector(settings).evaluate(crossed_pose())
    assert result.detected is True
    assert result.confidence >= 0.55
    assert CrossedArmDetector(settings).evaluate(normal_pose()).detected is False


def scaled_pose(scale: float, offset_x: float = 0.0) -> PoseLandmarks:
    pose = crossed_pose()

    def transform(point: Landmark) -> Landmark:
        return Landmark(0.5 + (point.x - 0.5) * scale + offset_x, 0.3 + (point.y - 0.3) * scale, visibility=point.visibility)

    return PoseLandmarks(*(transform(point) for point in pose.values()))


def hands_on_hips_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30), right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74), right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.25, 0.43), right_elbow=Landmark(0.75, 0.43),
        left_wrist=Landmark(0.32, 0.52), right_wrist=Landmark(0.68, 0.52),
    )


def raised_arms_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.26, 0.18),
        right_elbow=Landmark(0.74, 0.18),
        left_wrist=Landmark(0.18, 0.12),
        right_wrist=Landmark(0.82, 0.12),
    )


def arms_down_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.32, 0.62),
        right_elbow=Landmark(0.68, 0.62),
        left_wrist=Landmark(0.28, 0.82),
        right_wrist=Landmark(0.72, 0.82),
    )


def one_arm_cross_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.45, 0.42),
        right_elbow=Landmark(0.62, 0.42),
        left_wrist=Landmark(0.60, 0.38),
        right_wrist=Landmark(0.78, 0.58),
    )


def asymmetrical_cross_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.50, 0.48),
        right_elbow=Landmark(0.52, 0.38),
        left_wrist=Landmark(0.62, 0.40),
        right_wrist=Landmark(0.36, 0.44),
    )


def crossed_high_pose() -> PoseLandmarks:
    return PoseLandmarks(
        left_shoulder=Landmark(0.35, 0.30),
        right_shoulder=Landmark(0.65, 0.30),
        left_hip=Landmark(0.42, 0.74),
        right_hip=Landmark(0.58, 0.74),
        left_elbow=Landmark(0.38, 0.22),
        right_elbow=Landmark(0.62, 0.22),
        left_wrist=Landmark(0.58, 0.20),
        right_wrist=Landmark(0.42, 0.20),
    )


def test_normalized_features_are_stable_across_scale_and_offset():
    detector = CrossedArmDetector(GestureSettings(confidence_threshold=0.55))
    base = detector.evaluate(crossed_pose())
    scaled = detector.evaluate(scaled_pose(0.55, 0.12))
    assert scaled.detected is True
    assert scaled.confidence == pytest.approx(base.confidence, abs=0.02)


def test_false_positive_poses_are_rejected():
    detector = CrossedArmDetector(GestureSettings(confidence_threshold=0.55))
    assert detector.evaluate(hands_on_hips_pose()).detected is False


def test_partial_landmark_loss_does_not_activate():
    pose = crossed_pose()
    pose = PoseLandmarks(
        pose.left_shoulder, pose.right_shoulder, pose.left_elbow, pose.right_elbow,
        Landmark(pose.left_wrist.x, pose.left_wrist.y, visibility=0.1), pose.right_wrist,
    )
    assert CrossedArmDetector(GestureSettings(confidence_threshold=0.55)).evaluate(pose).detected is False


def test_state_machine_debounces_activation_release_and_cooldown():
    settings = GestureSettings(confirmation_frames=2, release_frames=2, cooldown_seconds=3.0)
    machine = GestureStateMachine(settings)
    positive = GestureResult(True, 0.9)
    negative = GestureResult(False, 0.1)
    assert machine.update(positive, 0) is GestureState.GESTURE_CANDIDATE
    assert machine.update(positive, 0.1) is GestureState.WAKANDA_ACTIVE
    assert machine.update(negative, 0.2) is GestureState.WAKANDA_ACTIVE
    assert machine.update(negative, 0.3) is GestureState.COOLDOWN
    assert machine.update(positive, 1.0) is GestureState.COOLDOWN
    assert machine.update(negative, 3.4) is GestureState.IDLE


def test_single_frame_and_low_confidence_do_not_activate():
    settings = GestureSettings(confirmation_frames=3, release_frames=2)
    machine = GestureStateMachine(settings)
    assert machine.update(GestureResult(True, 0.9), 0.0) is GestureState.GESTURE_CANDIDATE
    assert machine.update(GestureResult(False, 0.2), 0.1) is GestureState.GESTURE_CANDIDATE
    assert machine.update(GestureResult(False, 0.2), 0.2) is GestureState.IDLE


def test_held_pose_stays_active_without_retrigger_state():
    machine = GestureStateMachine(GestureSettings(confirmation_frames=2))
    positive = GestureResult(True, 0.9)
    assert machine.update(positive, 0.0) is GestureState.GESTURE_CANDIDATE
    assert machine.update(positive, 0.1) is GestureState.WAKANDA_ACTIVE
    assert all(machine.update(positive, 0.2 + index * 0.1) is GestureState.WAKANDA_ACTIVE for index in range(10))


def test_missing_pose_is_not_detected():
    result = CrossedArmDetector(GestureSettings()).evaluate(None)
    assert result == GestureResult(False, 0.0)


def test_crossed_arm_geometry_detects_valid_pose_and_rejects_raised_arms():
    detector = CrossedArmDetector(GestureSettings(confidence_threshold=0.55))
    assert detector.evaluate(crossed_pose()).detected is True
    assert detector.evaluate(raised_arms_pose()).detected is False
    assert detector.evaluate(arms_down_pose()).detected is False
    assert detector.evaluate(one_arm_cross_pose()).detected is False


def test_crossed_arms_at_different_heights_and_asymmetric_cases_are_supported():
    detector = CrossedArmDetector(GestureSettings(confidence_threshold=0.55))
    assert detector.evaluate(crossed_high_pose()).detected is True
    assert detector.evaluate(asymmetrical_cross_pose()).detected is True


def test_zero_byte_pose_model_is_rejected_with_clear_error(tmp_path):
    model_path = tmp_path / "pose_landmarker_lite.task"
    model_path.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="empty|zero-byte|empty file|not valid"):
        PoseDetector(PoseSettings(model_path=str(model_path)))
