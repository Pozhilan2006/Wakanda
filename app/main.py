from __future__ import annotations

import logging
import time

import cv2

from app.camera.webcam import Webcam
from app.config.settings import SettingsReloader, load_settings
from app.effects.audio import WakandaAudio
from app.effects.visual_effect import VisualEffect
from app.events.event_bus import EventBus
from app.performance import PerformanceMetrics, PerformanceTracker, process_resource_metrics
from app.vision.crossed_arm_detector import CrossedArmDetector, GestureState, GestureStateMachine
from app.vision.pose_detector import PoseDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def draw_debug(frame, pose, result, state, performance: PerformanceMetrics, activation_count: int, cooldown: float) -> None:
    height, width = frame.shape[:2]
    if pose is not None:
        points = {
            "left_shoulder": pose.left_shoulder,
            "right_shoulder": pose.right_shoulder,
            "left_elbow": pose.left_elbow,
            "right_elbow": pose.right_elbow,
            "left_wrist": pose.left_wrist,
            "right_wrist": pose.right_wrist,
        }
        for point in points.values():
            cv2.circle(frame, (int(point.x * width), int(point.y * height)), 6, (0, 220, 255), -1)
        connections = (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"))
        for first, second in connections:
            start, end = points[first], points[second]
            cv2.line(frame, (int(start.x * width), int(start.y * height)), (int(end.x * width), int(end.y * height)), (0, 180, 255), 3)
        cv2.line(frame, (int(pose.left_shoulder.x * width), int(pose.left_shoulder.y * height)), (int(pose.right_shoulder.x * width), int(pose.right_shoulder.y * height)), (255, 180, 0), 2)
        chest_x = int((pose.left_shoulder.x + pose.right_shoulder.x) * width / 2)
        chest_y = int((pose.left_shoulder.y + pose.right_shoulder.y) * height / 2)
        for wrist in (pose.left_wrist, pose.right_wrist):
            cv2.line(frame, (int(wrist.x * width), int(wrist.y * height)), (chest_x, chest_y), (255, 100, 0), 2)

    panel_width = min(490, width - 30)
    panel_height = 290
    panel = frame.copy()
    cv2.rectangle(panel, (15, 15), (15 + panel_width, 15 + panel_height), (15, 20, 30), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
    feature = result.features
    lines = [
        "WAKANDA GESTURE SYSTEM",
        f"FPS: {performance.fps:.1f}   Total latency: {performance.total_ms:.1f} ms",
        f"Pose: {'DETECTED' if pose else 'NOT DETECTED'}",
        f"Landmark quality: {feature.landmark_visibility:.2f}" if feature else "Landmark quality: 0.00",
        f"Gesture: {'CROSSED ARMS' if result.detected else 'NORMAL'}",
        f"Confidence: {result.confidence:.2f}",
        f"State: {state.value}",
        f"Activation count: {activation_count}",
        f"Cooldown: {cooldown:.1f} sec",
        f"Crossing: {feature.wrist_crossing:.2f}  Chest: {feature.chest_position:.2f}" if feature else "Crossing: 0.00  Chest: 0.00",
        f"Left/Right crossing: {feature.left_arm_crossing:.2f}/{feature.right_arm_crossing:.2f}" if feature else "Left/Right crossing: 0.00/0.00",
        f"Forearm intersection: {feature.arm_intersection:.2f}  Symmetry: {feature.arm_symmetry:.2f}" if feature else "Forearm intersection: 0.00  Symmetry: 0.00",
        f"Elbows: {feature.left_elbow_angle:.0f}/{feature.right_elbow_angle:.0f} deg" if feature else "Elbows: 0/0 deg",
        f"Pose: {performance.pose_ms:.1f} ms  Gesture: {performance.gesture_ms:.2f} ms",
    ]
    for index, text in enumerate(lines):
        color = (0, 220, 255) if index in (0, 4, 6) else (235, 235, 235)
        cv2.putText(frame, text, (28, 43 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    if feature:
        feature_lines = [
            f"Crossing {feature.wrist_crossing:.2f}  Chest {feature.chest_position:.2f}",
            f"Elbows {feature.left_elbow_angle:.0f}/{feature.right_elbow_angle:.0f} deg  Symmetry {feature.arm_symmetry:.2f}",
        ]
        for index, text in enumerate(feature_lines):
            cv2.putText(frame, text, (28, 330 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 190, 80), 2, cv2.LINE_AA)


def run() -> None:
    settings = load_settings()
    settings_reloader = SettingsReloader()
    bus = EventBus()
    audio = WakandaAudio(settings.audio)
    bus.subscribe("WAKANDA_ACTIVATED", audio.on_activated)
    bus.subscribe("WAKANDA_RELEASED", audio.on_released)
    detector = CrossedArmDetector(settings.gesture)
    machine = GestureStateMachine(settings.gesture)
    effect = VisualEffect()
    performance_tracker = PerformanceTracker()
    previous_state = machine.state
    activation_count = 0

    try:
        with Webcam(settings.camera) as webcam, PoseDetector(settings.pose) as pose_detector:
            for frame in webcam.frames():
                frame_start = time.perf_counter()
                updated = settings_reloader.poll()
                if updated is not None:
                    settings = updated
                    detector.settings = settings.gesture
                    machine.settings = settings.gesture
                    LOGGER.info("Reloaded gesture configuration from config.yaml")
                pose_start = time.perf_counter()
                pose = pose_detector.detect(frame)
                pose_ms = (time.perf_counter() - pose_start) * 1000
                gesture_start = time.perf_counter()
                result = detector.evaluate(pose)
                gesture_ms = (time.perf_counter() - gesture_start) * 1000
                now = time.time()
                state = machine.update(result, now)
                if state is GestureState.WAKANDA_ACTIVE and previous_state is not GestureState.WAKANDA_ACTIVE:
                    activation_count += 1
                    bus.publish("WAKANDA_ACTIVATED", confidence=result.confidence)
                    LOGGER.info("WAKANDA_ACTIVATED confidence=%.2f", result.confidence)
                elif previous_state is GestureState.WAKANDA_ACTIVE and state is GestureState.COOLDOWN:
                    bus.publish("WAKANDA_RELEASED")
                    LOGGER.info("WAKANDA_RELEASED")
                previous_state = state
                total_ms = (time.perf_counter() - frame_start) * 1000
                performance = performance_tracker.update(pose_ms, gesture_ms, total_ms)
                performance.cpu_percent, performance.memory_mb = process_resource_metrics()
                output = effect.render(frame, state is GestureState.WAKANDA_ACTIVE, result.confidence)
                draw_debug(output, pose, result, state, performance, activation_count, machine.cooldown_remaining(now))
                if performance.cpu_percent is not None:
                    cv2.putText(output, f"CPU {performance.cpu_percent:.0f}%  RAM {performance.memory_mb:.0f} MB", (28, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 2, cv2.LINE_AA)
                cv2.imshow("Wakanda Gesture Activation", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (RuntimeError, FileNotFoundError) as error:
        LOGGER.error("Unable to start prototype: %s", error)
    finally:
        audio.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
