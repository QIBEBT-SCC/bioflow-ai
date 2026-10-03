from bioflow_ai.schemas.notification import NotificationEvent, RuntimeNotificationChannel, WeComChannelConfig
from bioflow_ai.services.notification.providers.base import NotificationProviderClient
from bioflow_ai.services.notification.providers.wecom.client import WeComClient


class WeComNotificationProvider(NotificationProviderClient):
    def __init__(self, channel: RuntimeNotificationChannel) -> None:
        config = WeComChannelConfig.model_validate(channel.config)
        self._chat_id = config.chat_id
        self._client = WeComClient(bot_id=config.bot_id, secret=channel.secret, ws_url=config.ws_url)

    async def send(self, event: NotificationEvent) -> None:
        """Send one event message to the configured WeCom group."""
        await self._client.send_markdown(chat_id=self._chat_id, content=event.message)

    async def close(self) -> None:
        """Close the underlying raw WeCom client."""
        await self._client.close()
