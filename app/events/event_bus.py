from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from time import time
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    timestamp: float
    data: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "timestamp": self.timestamp, **self.data}


class EventBus:
    def __init__(self):
        self._subscribers: defaultdict[str, list[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, **data: Any) -> Event:
        event = Event(event_type, time(), data)
        for handler in tuple(self._subscribers[event_type]):
            handler(event)
        return event
