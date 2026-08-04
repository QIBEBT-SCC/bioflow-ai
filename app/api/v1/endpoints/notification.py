from typing import Annotated, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.auth import AdminDep
from app.core.session import SessionDep
from app.schemas.notification import (
    NOTIFICATION_TEST_EVENT,
    NotificationChannelCreate,
    NotificationChannelPublic,
    NotificationChannelUpdate,
    NotificationEventDefinition,
)
from app.services.notification import channels as channel_service
from app.services.notification.events import list_notification_events, publish_notification
from app.services.notification.gateway import NotificationGateway

router = APIRouter()


def _get_notification_gateway(request: Request) -> NotificationGateway:
    return request.app.state.notification_gateway


NotificationGatewayDep: TypeAlias = Annotated[NotificationGateway, Depends(_get_notification_gateway)]


@router.get('/events')
def get_notification_events(_admin: AdminDep) -> list[NotificationEventDefinition]:
    """Return every event type that can be subscribed to a channel."""
    return list_notification_events()


@router.get('/channels')
def list_notification_channels(
    session: SessionDep,
    _admin: AdminDep,
) -> list[NotificationChannelPublic]:
    """List configured notification channels."""
    return channel_service.list_channels(session)


@router.post('/channels', status_code=status.HTTP_201_CREATED)
async def create_notification_channel(
    data: NotificationChannelCreate,
    session: SessionDep,
    _admin: AdminDep,
    gateway: NotificationGatewayDep,
) -> NotificationChannelPublic:
    """Create a notification channel with an encrypted credential."""
    try:
        channel = channel_service.create_channel(session, data)
    except channel_service.NotificationChannelConflictError as error:
        raise HTTPException(status_code=409, detail='Notification channel name already exists.') from error
    except channel_service.UnsupportedNotificationProviderError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    await gateway.invalidate(channel.id)
    return channel


@router.patch('/channels/{channel_id}')
async def update_notification_channel(
    channel_id: int,
    data: NotificationChannelUpdate,
    session: SessionDep,
    _admin: AdminDep,
    gateway: NotificationGatewayDep,
) -> NotificationChannelPublic:
    """Update one notification channel."""
    try:
        channel = channel_service.update_channel(session, channel_id, data)
    except channel_service.NotificationChannelNotFoundError as error:
        raise HTTPException(status_code=404, detail='Notification channel not found.') from error
    except channel_service.NotificationChannelConflictError as error:
        raise HTTPException(status_code=409, detail='Notification channel name already exists.') from error
    await gateway.invalidate(channel.id)
    return channel


@router.delete('/channels/{channel_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification_channel(
    channel_id: int,
    session: SessionDep,
    _admin: AdminDep,
    gateway: NotificationGatewayDep,
) -> None:
    """Delete one notification channel."""
    try:
        channel_service.delete_channel(session, channel_id)
    except channel_service.NotificationChannelNotFoundError as error:
        raise HTTPException(status_code=404, detail='Notification channel not found.') from error
    await gateway.invalidate(channel_id)


@router.post('/channels/{channel_id}/test', status_code=status.HTTP_202_ACCEPTED)
async def send_test_notification(
    channel_id: int,
    session: SessionDep,
    _admin: AdminDep,
) -> dict[str, str]:
    """Queue a test message for one configured channel."""
    channel = next(
        (channel for channel in channel_service.list_channels(session) if channel.id == channel_id),
        None,
    )
    if channel is None:
        raise HTTPException(status_code=404, detail='Notification channel not found.')
    if not channel.enabled:
        raise HTTPException(status_code=409, detail='Enable the notification channel before testing it.')
    event = await publish_notification(
        NOTIFICATION_TEST_EVENT,
        source='api',
        message='## BioFlow AI 通知测试\n> 企业微信通知渠道已配置成功。\n\n这是一条测试消息，无需处理。',  # noqa: RUF001
        target_channel_id=channel_id,
    )
    return {'event_id': event.id, 'status': 'queued'}
