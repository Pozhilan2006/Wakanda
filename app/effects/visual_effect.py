from __future__ import annotations

import cv2
import numpy as np


class VisualEffect:
    def render(self, frame: np.ndarray, active: bool, confidence: float) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        if not active:
            return output
        overlay = output.copy()
        cv2.rectangle(overlay, (12, 12), (width - 12, height - 12), (0, 180, 255), 8)
        cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
        cv2.putText(output, "WAKANDA ACTIVATED", (width // 2 - 220, 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(output, f"POWER {confidence:.2f}", (width - 210, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        return output
