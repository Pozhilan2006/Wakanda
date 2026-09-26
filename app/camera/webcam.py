from __future__ import annotations

from collections.abc import Iterator
import sys

import cv2

from app.config.settings import CameraSettings


class Webcam:
    def __init__(self, settings: CameraSettings, device_index: int = 0):
        self.settings = settings
        self.device_index = device_index
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if sys.platform == "win32":
            capture = cv2.VideoCapture(
                self.device_index,
                cv2.CAP_DSHOW,
            )
        else:
            capture = cv2.VideoCapture(self.device_index)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open webcam index {self.device_index}"
            )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.settings.width,
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.settings.height,
        )
        capture.set(
            cv2.CAP_PROP_FPS,
            self.settings.fps,
        )

        # Verify that the camera can actually provide a frame.
        success, frame = capture.read()

        if not success or frame is None:
            capture.release()
            raise RuntimeError(
                f"Unable to capture frames from webcam "
                f"index {self.device_index}"
            )

        self.capture = capture

    def frames(self) -> Iterator[object]:
        if self.capture is None:
            self.open()

        assert self.capture is not None

        while True:
            success, frame = self.capture.read()

            if not success:
                raise RuntimeError(
                    "Webcam frame capture failed"
                )

            yield frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> Webcam:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()