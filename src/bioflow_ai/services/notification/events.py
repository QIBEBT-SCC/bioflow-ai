import uuid

from redis.asyncio import Redis

from bioflow_ai.core.redis_client import notification_redis_client
from bioflow_ai.schemas.notification import NotificationEvent, NotificationEventDefinition, NotificationEventType
from bioflow_ai.utils.system import utc_now

NOTIFICATION_STREAM = 'bioflow:notifications'
NOTIFICATION_STREAM_MAX_LENGTH = 10_000
NOTIFICATION_EVENT_TTL_SECONDS = 60

NOTIFICATION_EVENT_CATALOG = (
    NotificationEventDefinition(
        key=NotificationEventType.TOOL_CREATE_SUCCEEDED,
        name={
            'zh': '工具创建成功',
            'en': 'Tool creation succeeded',
        },
        description={
            'zh': '每成功创建一个工具，向已订阅的 Channel 发送一条通知。',  # noqa: RUF001
            'en': 'Send one notification to subscribed channels for every successfully created tool.',
        },
    ),
    NotificationEventDefinition(
        key=NotificationEventType.TOOL_CREATE_INTERRUPTED,
        name={
            'zh': '工具创建被中断',
            'en': 'Tool creation interrupted',
        },
        description={
            'zh': '工具创建流程进入人工确认或其他中断状态时发送通知。',
            'en': 'Notify when tool creation stops for human review or another interrupt.',
        },
    ),
    NotificationEventDefinition(
        key=NotificationEventType.TOOL_CREATE_FAILED,
        name={
            'zh': '工具创建异常退出',
            'en': 'Tool creation failed',
        },
        description={
            'zh': '工具创建过程中发生未处理异常并退出时发送通知。',
            'en': 'Notify when tool creation exits because of an unhandled error.',
        },
    ),
)


def list_notification_events() -> list[NotificationEventDefinition]:
    """Return the notification event catalog exposed to configuration clients."""
    return list(NOTIFICATION_EVENT_CATALOG)


def create_notification_event(
    event_type: str,
    *,
    source: str,
    message: str,
    target_channel_id: int | None = None,
) -> NotificationEvent:
    """Create the provider-neutral event envelope used by every producer."""
    return NotificationEvent(
        id=uuid.uuid4().hex,
        type=event_type,
        source=source,
        occurred_at=utc_now(),
        message=message,
        target_channel_id=target_channel_id,
    )


async def publish_notification(
    event_type: str,
    *,
    source: str,
    message: str,
    target_channel_id: int | None = None,
    redis: Redis = notification_redis_client,
) -> NotificationEvent:
    """Create and publish an outbound notification event."""
    event = create_notification_event(
        event_type,
        source=source,
        message=message,
        target_channel_id=target_channel_id,
    )
    await redis.xadd(
        NOTIFICATION_STREAM,
        {'payload': event.model_dump_json()},
        maxlen=NOTIFICATION_STREAM_MAX_LENGTH,
        approximate=True,
    )
    return event
