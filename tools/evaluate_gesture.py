from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import GestureSettings, load_settings
from app.vision.crossed_arm_detector import CrossedArmDetector, GestureResult, GestureState, GestureStateMachine, PoseLandmarks
from app.vision.landmark_utils import Landmark


@dataclass(frozen=True)
class Metrics:
    samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    fps: float
    latency_ms: float


def pose_from_record(record: dict) -> PoseLandmarks:
    return PoseLandmarks(**{
        name: Landmark(**record["landmarks"][name])
        for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    })


def basic_rules(pose: PoseLandmarks, settings: GestureSettings) -> bool:
    shoulder_width = max(((pose.right_shoulder.x - pose.left_shoulder.x) ** 2 + (pose.right_shoulder.y - pose.left_shoulder.y) ** 2) ** 0.5, 1e-6)
    chest_x = (pose.left_shoulder.x + pose.right_shoulder.x) / 2.0
    chest_y = (pose.left_shoulder.y + pose.right_shoulder.y) / 2.0
    wrist_distance = (
        ((pose.left_wrist.x - chest_x) ** 2 + (pose.left_wrist.y - chest_y) ** 2) ** 0.5
        + ((pose.right_wrist.x - chest_x) ** 2 + (pose.right_wrist.y - chest_y) ** 2) ** 0.5
    ) / (2.0 * shoulder_width)
    return pose.left_wrist.x > chest_x and pose.right_wrist.x < chest_x and wrist_distance <= settings.wrist_chest_distance


def classify(records: list[dict], method: str, settings: GestureSettings) -> tuple[list[bool], float, float]:
    detector = CrossedArmDetector(settings)
    predictions: list[bool] = []
    durations: list[float] = []
    machine = GestureStateMachine(settings)
    current_session = None
    for record in records:
        if method == "temporal_normalized" and record.get("session_id") != current_session:
            machine = GestureStateMachine(settings)
            current_session = record.get("session_id")
        pose = pose_from_record(record)
        started = time.perf_counter()
        result = detector.evaluate(pose)
        if method == "basic_rules":
            prediction = basic_rules(pose, settings)
        elif method == "normalized_rules":
            prediction = result.detected
        else:
            timestamp = float(record.get("timestamp", time.time()))
            prediction = machine.update(result, timestamp) is GestureState.WAKANDA_ACTIVE
        durations.append(time.perf_counter() - started)
        predictions.append(prediction)
    average = sum(durations) / len(durations) if durations else 0.0
    return predictions, average * 1000.0, 1.0 / average if average else 0.0


def calculate_metrics(records: list[dict], predictions: list[bool], latency_ms: float, fps: float) -> Metrics:
    actual = [record["label"] == "CROSSED_ARMS" for record in records]
    true_positive = sum(expected and predicted for expected, predicted in zip(actual, predictions))
    true_negative = sum(not expected and not predicted for expected, predicted in zip(actual, predictions))
    false_positive = sum(not expected and predicted for expected, predicted in zip(actual, predictions))
    false_negative = sum(expected and not predicted for expected, predicted in zip(actual, predictions))
    samples = len(records)
    accuracy = (true_positive + true_negative) / samples if samples else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0
    false_negative_rate = false_negative / (false_negative + true_positive) if false_negative + true_positive else 0.0
    return Metrics(samples, accuracy, precision, recall, f1, false_positive_rate, false_negative_rate, true_positive, true_negative, false_positive, false_negative, fps, latency_ms)


def print_report(results: dict[str, Metrics]) -> None:
    samples = next(iter(results.values())).samples if results else 0
    print("Wakanda Gesture Evaluation")
    print("--------------------------")
    print(f"Samples: {samples}")
    for name, metrics in results.items():
        print(f"\n{name}")
        print(f"Accuracy: {metrics.accuracy * 100:.2f}%")
        print(f"Precision: {metrics.precision * 100:.2f}%")
        print(f"Recall: {metrics.recall * 100:.2f}%")
        print(f"F1 Score: {metrics.f1 * 100:.2f}%")
        print(f"False Positive Rate: {metrics.false_positive_rate * 100:.2f}%")
        print(f"False Negative Rate: {metrics.false_negative_rate * 100:.2f}%")
        print(f"Confusion matrix: TP={metrics.true_positive} TN={metrics.true_negative} FP={metrics.false_positive} FN={metrics.false_negative}")
        print(f"Evaluation FPS: {metrics.fps:.1f}")
        print(f"Gesture latency: {metrics.latency_ms:.3f} ms")
    print("\nBenchmark")
    print("Method | Accuracy | Precision | Recall | F1 | FPS | Latency (ms)")
    for name, metrics in results.items():
        print(f"{name} | {metrics.accuracy * 100:.2f}% | {metrics.precision * 100:.2f}% | {metrics.recall * 100:.2f}% | {metrics.f1 * 100:.2f}% | {metrics.fps:.1f} | {metrics.latency_ms:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Wakanda crossed-arm strategies on collected JSONL data.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("Dataset is empty; no results were generated.")
    settings = load_settings(args.config).gesture
    results = {}
    for method in ("basic_rules", "normalized_rules", "temporal_normalized"):
        predictions, latency_ms, fps = classify(records, method, settings)
        results[method] = calculate_metrics(records, predictions, latency_ms, fps)
    print_report(results)


if __name__ == "__main__":
    main()
