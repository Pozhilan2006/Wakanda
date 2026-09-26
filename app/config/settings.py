from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraSettings:
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class PoseSettings:
    model_path: str = "assets/models/pose_landmarker_lite.task"
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class GestureSettings:
    confirmation_frames: int = 8
    release_frames: int = 8
    cooldown_seconds: float = 3.0
    confidence_threshold: float = 0.75
    visibility_threshold: float = 0.55
    wrist_chest_distance: float = 0.75
    crossing_margin: float = 0.08
    chest_vertical_min: float = -0.25
    chest_vertical_max: float = 0.75
    wrist_crossing_min: float = 0.35
    chest_position_min: float = 0.20
    elbow_angle_min: float = 55.0
    elbow_angle_max: float = 175.0
    pose_motion_limit: float = 1.5
    weights: dict[str, float] = field(default_factory=lambda: {
        "crossing": 0.35,
        "chest_occupancy": 0.30,
        "forearm_intersection": 0.20,
        "elbow_geometry": 0.08,
        "arm_symmetry": 0.05,
        "landmark_visibility": 0.02,
    })


@dataclass(frozen=True)
class AudioSettings:
    wakanda_file: str = "assets/audio/wakanda_bgm.wav"
    volume: float = 0.75


@dataclass(frozen=True)
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    pose: PoseSettings = field(default_factory=PoseSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_settings(path: str | Path = "config.yaml") -> Settings:
    config_path = Path(path)
    if not config_path.exists():
        return Settings()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Settings(
        camera=CameraSettings(**_section(data, "camera")),
        pose=PoseSettings(**_section(data, "pose")),
        gesture=GestureSettings(**_section(data, "gesture")),
        audio=AudioSettings(**_section(data, "audio")),
    )


class SettingsReloader:
    def __init__(self, path: str | Path = "config.yaml"):
        self.path = Path(path)
        self._modified = self.path.stat().st_mtime if self.path.exists() else None

    def poll(self) -> Settings | None:
        modified = self.path.stat().st_mtime if self.path.exists() else None
        if modified == self._modified:
            return None
        self._modified = modified
        return load_settings(self.path)
