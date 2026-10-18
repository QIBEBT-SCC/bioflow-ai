import httpx

from bioflow_ai.schemas.notification import NotificationEvent, RuntimeNotificationChannel

DISCORD_MESSAGE_MAX_LENGTH = 2000


class DiscordWebhookError(RuntimeError):
    """Raised when Discord rejects a webhook message."""


class DiscordNotificationProvider:
    """Deliver notification events through one Discord incoming webhook."""

    def __init__(self, channel: RuntimeNotificationChannel) -> None:
        self._webhook_url = channel.secret
        self._client = httpx.AsyncClient(timeout=20)

    async def send(self, event: NotificationEvent) -> None:
        """Post one event message to the webhook channel."""
        response = await self._client.post(
            self._webhook_url,
            params={'wait': 'true'},
            json={'content': event.message[:DISCORD_MESSAGE_MAX_LENGTH], 'allowed_mentions': {'parse': []}},
        )
        if response.is_error:
            # The webhook URL is the credential, so errors only carry the status and Discord's reply.
            message = f'Discord webhook returned HTTP {response.status_code}: {response.text[:300]}'
            raise DiscordWebhookError(message)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
