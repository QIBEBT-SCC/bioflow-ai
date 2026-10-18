from typing import Protocol

from bioflow_ai.schemas.notification import NotificationEvent, NotificationProvider, RuntimeNotificationChannel
from bioflow_ai.services.notification.providers.wecom.provider import WeComNotificationProvider


class NotificationProviderClient(Protocol):
    """Provider boundary used by the notification gateway."""

    async def send(self, event: NotificationEvent) -> None:
        """Send one notification event."""
        ...

    async def close(self) -> None:
        """Release provider resources."""
        ...


def build_provider(channel: RuntimeNotificationChannel) -> NotificationProviderClient:
    """Build the provider adapter required by one runtime channel."""
    if channel.provider is NotificationProvider.WECOM:
        return WeComNotificationProvider(channel)
    msg = f'Provider {channel.provider.value!r} is not implemented.'
    raise ValueError(msg)
