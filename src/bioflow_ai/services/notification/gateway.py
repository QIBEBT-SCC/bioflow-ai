import asyncio
import contextlib
from datetime import datetime, timedelta

from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy.orm import Session

import bioflow_ai.crud.notification.channel as channel_crud
from bioflow_ai.infrastructure.database import engine
from bioflow_ai.schemas.notification import NotificationEvent, RuntimeNotificationChannel
from bioflow_ai.services.notification.channels import runtime_channels_for_event
from bioflow_ai.services.notification.events import NOTIFICATION_EVENT_TTL_SECONDS, NOTIFICATION_STREAM
from bioflow_ai.services.notification.providers import NotificationProviderClient, build_provider
from bioflow_ai.utils.system import utc_now

NOTIFICATION_CONSUMER_GROUP = 'notification-gateway'
NOTIFICATION_CONSUMER_NAME = 'server'


class NotificationGateway:
    """Consume provider-neutral events and dispatch them through configured channels."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._providers: dict[int, tuple[str, NotificationProviderClient]] = {}
        self._runner: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the consumer group and start the background dispatcher."""
        try:
            await self._redis.xgroup_create(
                NOTIFICATION_STREAM,
                NOTIFICATION_CONSUMER_GROUP,
                id='0',
                mkstream=True,
            )
        except ResponseError as error:
            if 'BUSYGROUP' not in str(error):
                raise
        self._runner = asyncio.create_task(self._run(), name='notification-gateway')

    async def close(self) -> None:
        """Stop dispatch and close all provider connections."""
        if self._runner is not None:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
        providers, self._providers = self._providers, {}
        await asyncio.gather(
            *(provider.close() for _, provider in providers.values()),
            return_exceptions=True,
        )

    async def invalidate(self, channel_id: int) -> None:
        """Drop a cached provider after a channel configuration change."""
        cached = self._providers.pop(channel_id, None)
        if cached is not None:
            await cached[1].close()

    async def _run(self) -> None:
        while True:
            try:
                messages = await self._read_messages('0', block=None)
                if not messages:
                    messages = await self._read_messages('>', block=5_000)
                for message_id, fields in messages:
                    await self._dispatch(message_id, fields)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception('Notification gateway iteration failed; retrying.')
                await asyncio.sleep(5)

    async def _read_messages(self, offset: str, *, block: int | None) -> list[tuple[str, dict[str, str]]]:
        streams = await self._redis.xreadgroup(
            NOTIFICATION_CONSUMER_GROUP,
            NOTIFICATION_CONSUMER_NAME,
            {NOTIFICATION_STREAM: offset},
            count=1,
            block=block,
        )
        if not streams:
            return []
        return streams[0][1]

    async def _dispatch(self, message_id: str, fields: dict[str, str]) -> None:
        event = NotificationEvent.model_validate_json(fields['payload'])
        if self._event_expired(event):
            logger.info('Discarding expired notification event {}.', event.id)
            await self._finish(message_id, event.id)
            return

        channels = await asyncio.to_thread(self._load_channels, event)
        if not channels:
            await self._finish(message_id, event.id)
            return

        for channel in channels:
            if self._event_expired(event):
                logger.info('Notification event {} expired before all channels were delivered.', event.id)
                await self._finish(message_id, event.id)
                return
            delivery_key = f'bioflow:notification:delivered:{event.id}'
            channel_key = str(channel.id)
            if await self._redis.execute_command('SISMEMBER', delivery_key, channel_key):
                continue
            attempted_at = utc_now()
            try:
                provider = await self._provider_for(channel)
                await provider.send(event)
            except Exception as error:
                error_message = f'{type(error).__name__}: {error}'
                await self._record_delivery_result(
                    channel.id,
                    attempted_at=attempted_at,
                    succeeded_at=None,
                    error=error_message,
                )
                logger.exception(
                    'Notification delivery failed for event {} on channel {}; it will be retried.',
                    event.id,
                    channel.id,
                )
                if self._event_expired(event):
                    await self._finish(message_id, event.id)
                    return
                raise
            await self._redis.execute_command('SADD', delivery_key, channel_key)
            await self._redis.expire(delivery_key, NOTIFICATION_EVENT_TTL_SECONDS)
            await self._record_delivery_result(
                channel.id,
                attempted_at=attempted_at,
                succeeded_at=utc_now(),
                error=None,
            )
        await self._finish(message_id, event.id)

    async def _provider_for(self, channel: RuntimeNotificationChannel) -> NotificationProviderClient:
        cached = self._providers.get(channel.id)
        if cached is not None:
            version, provider = cached
            if version == channel.version:
                return provider
            await provider.close()
        provider = build_provider(channel)
        self._providers[channel.id] = (channel.version, provider)
        return provider

    async def _finish(self, message_id: str, event_id: str) -> None:
        """Acknowledge and immediately delete a terminal notification event."""
        await self._redis.xack(NOTIFICATION_STREAM, NOTIFICATION_CONSUMER_GROUP, message_id)
        await self._redis.xdel(NOTIFICATION_STREAM, message_id)
        await self._redis.delete(f'bioflow:notification:delivered:{event_id}')

    async def _record_delivery_result(
        self,
        channel_id: int,
        *,
        attempted_at: datetime,
        succeeded_at: datetime | None,
        error: str | None,
    ) -> None:
        def save() -> None:
            with Session(engine) as session:
                channel_crud.record_delivery_result(
                    session,
                    channel_id,
                    attempted_at=attempted_at,
                    succeeded_at=succeeded_at,
                    error=error,
                )

        try:
            await asyncio.to_thread(save)
        except Exception:
            logger.exception('Failed to persist notification delivery status for channel {}.', channel_id)

    @staticmethod
    def _event_expired(event: NotificationEvent) -> bool:
        return utc_now() - event.occurred_at >= timedelta(seconds=NOTIFICATION_EVENT_TTL_SECONDS)

    @staticmethod
    def _load_channels(event: NotificationEvent) -> list[RuntimeNotificationChannel]:
        with Session(engine) as session:
            return runtime_channels_for_event(session, event.type, event.target_channel_id)
