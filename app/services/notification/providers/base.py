from abc import ABC, abstractmethod

from app.schemas.notification import NotificationEvent


class NotificationProviderClient(ABC):
    """Small provider boundary shared by WeCom and future Discord/Telegram clients."""

    @abstractmethod
    async def send(self, event: NotificationEvent) -> None:
        """Send one notification event."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
