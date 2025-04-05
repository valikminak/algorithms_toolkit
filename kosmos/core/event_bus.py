# kosmos/core/event_bus.py
from typing import Dict, List, Callable, Any


class EventBus:
    """
    A central event bus that allows components to communicate without
    direct dependencies.
    """

    def __init__(self):
        self.listeners: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event with optional data."""
        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                callback(data)

    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Remove a subscription."""
        if event_type in self.listeners and callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)