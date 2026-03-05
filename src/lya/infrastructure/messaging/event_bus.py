"""Event Bus Implementation.

In-memory event bus for publishing and subscribing to domain events.
"""

from typing import Callable, TypeVar
import asyncio

from lya.domain.models.events import DomainEvent, EventPublisher


T = TypeVar("T", bound=DomainEvent)


class InMemoryEventBus(EventPublisher):
    """
    In-memory implementation of the event bus.

    For production, this could be replaced with Redis, RabbitMQ, etc.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[Callable]] = {}
        self._history: list[DomainEvent] = []

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        self._history.append(event)

        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        # Run handlers concurrently
        if handlers:
            await asyncio.gather(
                *[self._run_handler(handler, event) for handler in handlers],
                return_exceptions=True
            )

    async def _run_handler(self, handler: Callable, event: DomainEvent) -> None:
        """Run a handler, catching exceptions."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            # Log but don't stop other handlers
            print(f"Event handler failed: {e}")

    def get_history(self) -> list[DomainEvent]:
        """Get event history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
