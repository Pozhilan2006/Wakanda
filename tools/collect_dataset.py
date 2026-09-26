from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from app.camera.webcam import Webcam
from app.config.settings import load_settings
from app.vision.crossed_arm_detector import CrossedArmDetector
from app.vision.pose_detector import PoseDetector


def collect(label: str, user_id: str, session_id: str, output_path: Path, max_samples: int | None) -> int:
    settings = load_settings()
    detector = CrossedArmDetector(settings.gesture)
    samples = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        with Webcam(settings.camera) as webcam, PoseDetector(settings.pose) as pose_detector:
            for frame in webcam.frames():
                pose = pose_detector.detect(frame)
                if pose is not None:
                    result = detector.evaluate(pose)
                    record = {
                        "timestamp": time.time(),
                        "user_id": user_id,
                        "session_id": session_id,
                        "label": label,
                        "landmarks": pose.as_dict(),
                        "features": result.features.as_dict() if result.features else {},
                        "confidence": result.confidence,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    samples += 1
                cv2.putText(frame, f"LABEL: {label}  SAMPLES: {samples}  Q: quit", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                cv2.imshow("Wakanda Dataset Collection", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q") or (max_samples and samples >= max_samples):
                    break
    cv2.destroyAllWindows()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect normalized Wakanda gesture samples; raw video is not stored.")
    parser.add_argument("--label", choices=("CROSSED_ARMS", "NOT_CROSSED_ARMS"), required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--output", type=Path, default=Path("data/gesture_dataset.jsonl"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    session_id = args.session_id or str(int(time.time()))
    print(f"Collecting {args.label} for user {args.user_id}, session {session_id}")
    print(f"Writing normalized landmarks to {args.output}")
    print(f"Collected {collect(args.label, args.user_id, session_id, args.output, args.max_samples)} samples")


if __name__ == "__main__":
    main()
