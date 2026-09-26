from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import PoseSettings
from app.vision.crossed_arm_detector import PoseLandmarks
from app.vision.landmark_utils import Landmark


POSE_LANDMARKER_MODEL_SOURCE = "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python"


class PoseDetector:
    """Small adapter around MediaPipe's current Tasks Pose Landmarker API."""

    def __init__(self, settings: PoseSettings):
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                "MediaPipe Pose Landmarker model is missing. "
                f"Expected file: {model_path.name}; expected location: {model_path}. "
                f"Download the official Pose Landmarker model from {POSE_LANDMARKER_MODEL_SOURCE} "
                f"and place the file at {model_path}."
            )
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise FileNotFoundError(
                "MediaPipe Pose Landmarker model file is empty or invalid. "
                f"Expected a non-empty file at {model_path}. "
                f"Download the official Pose Landmarker model from {POSE_LANDMARKER_MODEL_SOURCE} "
                f"and replace the invalid file at {model_path}."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as error:
            raise RuntimeError("MediaPipe is not installed; run pip install -r requirements.txt") from error

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=settings.min_pose_detection_confidence,
            min_pose_presence_confidence=settings.min_pose_presence_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self._vision = vision
        self._mp = mp
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> PoseLandmarks | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]

        def landmark(index: int) -> Landmark:
            point = landmarks[index]
            return Landmark(point.x, point.y, point.z, getattr(point, "visibility", 1.0))

        # MediaPipe Pose indices: shoulders 11/12, hips 23/24, elbows 13/14, wrists 15/16.
        return PoseLandmarks(
            left_shoulder=landmark(11),
            right_shoulder=landmark(12),
            left_hip=landmark(23),
            right_hip=landmark(24),
            left_elbow=landmark(13),
            right_elbow=landmark(14),
            left_wrist=landmark(15),
            right_wrist=landmark(16),
        )

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
