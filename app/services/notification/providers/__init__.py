from app.schemas.notification import NotificationProvider, RuntimeNotificationChannel
from app.services.notification.providers.base import NotificationProviderClient
from app.services.notification.providers.wecom.provider import WeComNotificationProvider


def build_provider(channel: RuntimeNotificationChannel) -> NotificationProviderClient:
    """Build the provider adapter required by one runtime channel."""
    if channel.provider is NotificationProvider.WECOM:
        return WeComNotificationProvider(channel)
    msg = f'Provider {channel.provider.value!r} is not implemented.'
    raise ValueError(msg)


__all__ = ['NotificationProviderClient', 'WeComNotificationProvider', 'build_provider']
