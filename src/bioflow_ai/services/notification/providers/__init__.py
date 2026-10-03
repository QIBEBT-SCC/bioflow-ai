from bioflow_ai.schemas.notification import NotificationProvider, RuntimeNotificationChannel
from bioflow_ai.services.notification.providers.base import NotificationProviderClient
from bioflow_ai.services.notification.providers.wecom.provider import WeComNotificationProvider


def build_provider(channel: RuntimeNotificationChannel) -> NotificationProviderClient:
    """Build the provider adapter required by one runtime channel."""
    if channel.provider is NotificationProvider.WECOM:
        return WeComNotificationProvider(channel)
    msg = f'Provider {channel.provider.value!r} is not implemented.'
    raise ValueError(msg)


__all__ = ['NotificationProviderClient', 'WeComNotificationProvider', 'build_provider']
