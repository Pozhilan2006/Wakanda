from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import AudioSettings
from app.events.event_bus import Event

LOGGER = logging.getLogger(__name__)


class WakandaAudio:
    def __init__(self, settings: AudioSettings):
        self.path = Path(settings.wakanda_file)
        self.volume = settings.volume
        self._available = False
        try:
            import pygame

            pygame.mixer.init()
            self._pygame = pygame
            if self.path.exists():
                pygame.mixer.music.load(str(self.path))
                pygame.mixer.music.set_volume(self.volume)
                self._available = True
            else:
                LOGGER.warning("Audio file not found: %s", self.path)
        except (ImportError, RuntimeError, OSError) as error:
            self._pygame = None
            LOGGER.warning("Audio unavailable: %s", error)

    def on_activated(self, _: Event) -> None:
        if self._available and self._pygame is not None:
            self._pygame.mixer.music.play(loops=0)

    def on_released(self, _: Event) -> None:
        if self._available and self._pygame is not None:
            self._pygame.mixer.music.fadeout(400)

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.mixer.quit()
