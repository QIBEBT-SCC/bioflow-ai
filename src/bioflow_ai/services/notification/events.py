import uuid
from collections.abc import Mapping
from datetime import timedelta

from loguru import logger
from redis.asyncio import Redis

from bioflow_ai.core.config import get_settings
from bioflow_ai.schemas.notification import NotificationEvent, NotificationEventDefinition, NotificationEventType
from bioflow_ai.utils.system import utc_now

NOTIFICATION_STREAM = 'bioflow:notifications'
NOTIFICATION_STREAM_MAX_LENGTH = 10_000
NOTIFICATION_EVENT_TTL_SECONDS = 60


def _event(
    key: NotificationEventType, name: tuple[str, str], description: tuple[str, str]
) -> NotificationEventDefinition:
    return NotificationEventDefinition(
        key=key,
        name={'zh': name[0], 'en': name[1]},
        description={'zh': description[0], 'en': description[1]},
    )


NOTIFICATION_EVENT_CATALOG = [
    _event(
        NotificationEventType.TOOL_CREATE_SUCCEEDED,
        ('工具创建成功', 'Tool creation succeeded'),
        (
            '每成功创建一个工具，向已订阅的 Channel 发送一条通知。',  # noqa: RUF001
            'Send one notification to subscribed channels for every successfully created tool.',
        ),
    ),
    _event(
        NotificationEventType.TOOL_CREATE_INTERRUPTED,
        ('工具创建被中断', 'Tool creation interrupted'),
        (
            '工具创建流程进入人工确认或其他中断状态时发送通知。',
            'Notify when tool creation stops for human review or another interrupt.',
        ),
    ),
    _event(
        NotificationEventType.TOOL_CREATE_FAILED,
        ('工具创建异常退出', 'Tool creation failed'),
        (
            '工具创建过程中发生未处理异常并退出时发送通知。',
            'Notify when tool creation exits because of an unhandled error.',
        ),
    ),
    _event(
        NotificationEventType.WORKFLOW_BUILD_SUCCEEDED,
        ('工作流构建成功', 'Workflow build succeeded'),
        (
            'Workflow Builder 完成工作流构建时发送通知。',
            'Notify when the Workflow Builder finishes building a workflow.',
        ),
    ),
    _event(
        NotificationEventType.WORKFLOW_BUILD_INTERRUPTED,
        ('工作流构建被中断', 'Workflow build interrupted'),
        (
            'Workflow Builder 进入人工确认或其他中断状态时发送通知。',
            'Notify when the Workflow Builder stops for human review or another interrupt.',
        ),
    ),
    _event(
        NotificationEventType.WORKFLOW_BUILD_FAILED,
        ('工作流构建异常退出', 'Workflow build failed'),
        (
            'Workflow Builder 运行过程中发生异常并退出时发送通知。',
            'Notify when the Workflow Builder exits because of an error.',
        ),
    ),
]


def format_duration(duration: timedelta) -> str:
    """Format an elapsed time for notification messages."""
    minutes, seconds = divmod(max(0, round(duration.total_seconds())), 60)
    return f'{minutes} min {seconds} sec' if minutes else f'{seconds} sec'


async def publish_notification(
    event_type: str,
    *,
    source: str,
    message: str,
    target_channel_id: int | None = None,
) -> NotificationEvent:
    """Create and publish an outbound notification event.

    A short-lived client is used so producers running under separate event loops, such as Celery tasks, never
    reuse connections bound to another loop.
    """
    event = NotificationEvent(
        id=uuid.uuid4().hex,
        type=event_type,
        source=source,
        occurred_at=utc_now(),
        message=message,
        target_channel_id=target_channel_id,
    )
    url = get_settings().persistent.cache.NOTIFICATION_CACHE.unicode_string()
    async with Redis.from_url(url, decode_responses=True, socket_timeout=5) as redis:
        await redis.xadd(
            NOTIFICATION_STREAM,
            {'payload': event.model_dump_json()},
            maxlen=NOTIFICATION_STREAM_MAX_LENGTH,
            approximate=True,
        )
    return event


async def notify(
    event_type: str,
    *,
    source: str,
    title: str,
    fields: Mapping[str, object],
    note: str | None = None,
) -> None:
    """Publish a Markdown event for subscribed channels without affecting the producer's outcome."""
    lines = [f'## BioFlow AI {title}', *(f'- {key}: {value}' for key, value in fields.items())]
    if note:
        lines.append(f'> {note[:1000]}')
    try:
        await publish_notification(event_type, source=source, message='\n'.join(lines))
    except Exception:
        logger.exception('Failed to publish the {} notification event.', event_type)
