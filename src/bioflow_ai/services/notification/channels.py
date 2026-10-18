from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import bioflow_ai.crud.notification.channel as channel_crud
from bioflow_ai.core.secret_cipher import decrypt_secret, encrypt_secret
from bioflow_ai.models.notification import NotificationChannel, NotificationSubscription
from bioflow_ai.schemas.notification import (
    NotificationChannelCreate,
    NotificationChannelPublic,
    NotificationChannelUpdate,
    NotificationEventType,
    NotificationProvider,
    RuntimeNotificationChannel,
    WeComChannelConfig,
)
from bioflow_ai.utils.system import utc_now

NOTIFICATION_CREDENTIAL_PURPOSE = 'notification-credentials'


class NotificationChannelError(ValueError):
    """Base error for notification-channel operations."""


class NotificationChannelNotFoundError(NotificationChannelError):
    """Raised when a configured notification channel does not exist."""


class NotificationChannelConflictError(NotificationChannelError):
    """Raised when a channel name already exists."""


class UnsupportedNotificationProviderError(NotificationChannelError):
    """Raised when a provider is known but not implemented yet."""


def _validate_config(provider: NotificationProvider, config: dict) -> dict:
    if provider is NotificationProvider.WECOM:
        return WeComChannelConfig.model_validate(config).model_dump()
    message = f'Provider {provider.value!r} is not implemented yet.'
    raise UnsupportedNotificationProviderError(message)


def _set_subscriptions(channel: NotificationChannel, event_types: list[NotificationEventType]) -> None:
    desired = list(dict.fromkeys(event_type.value for event_type in event_types))
    kept = [subscription for subscription in channel.subscriptions if subscription.event_type in desired]
    existing = {subscription.event_type for subscription in kept}
    channel.subscriptions = kept + [
        NotificationSubscription(event_type=event_type) for event_type in desired if event_type not in existing
    ]


def to_public(channel: NotificationChannel) -> NotificationChannelPublic:
    """Project a stored channel without exposing encrypted credentials."""
    return NotificationChannelPublic(
        id=channel.id,
        name=channel.name,
        provider=NotificationProvider(channel.provider),
        enabled=channel.enabled,
        config=channel.config,
        credential_configured=bool(channel.secret_ciphertext),
        event_types=[subscription.event_type for subscription in channel.subscriptions],
        last_attempt_at=channel.last_attempt_at,
        last_success_at=channel.last_success_at,
        last_error=channel.last_error,
        created_at=channel.created_at,
        updated_at=channel.updated_at,
    )


def get_channel(session: Session, channel_id: int) -> NotificationChannel:
    """Return one stored channel or raise when it does not exist."""
    channel = channel_crud.get(session, channel_id)
    if channel is None:
        raise NotificationChannelNotFoundError(channel_id)
    return channel


def create_channel(session: Session, data: NotificationChannelCreate) -> NotificationChannelPublic:
    """Validate, encrypt, and store one notification channel."""
    config = _validate_config(data.provider, data.config)
    channel = NotificationChannel(
        name=data.name,
        provider=data.provider.value,
        enabled=data.enabled,
        config=config,
        secret_ciphertext=encrypt_secret(
            data.secret.get_secret_value(),
            purpose=NOTIFICATION_CREDENTIAL_PURPOSE,
        ),
    )
    _set_subscriptions(channel, data.event_types)
    try:
        channel_crud.insert(session, channel)
    except IntegrityError as error:
        session.rollback()
        raise NotificationChannelConflictError(data.name) from error
    return to_public(channel)


def update_channel(
    session: Session,
    channel_id: int,
    data: NotificationChannelUpdate,
) -> NotificationChannelPublic:
    """Update a notification channel while preserving an omitted secret."""
    channel = get_channel(session, channel_id)
    if data.name is not None:
        channel.name = data.name
    if data.enabled is not None:
        channel.enabled = data.enabled
    if data.config is not None:
        channel.config = _validate_config(NotificationProvider(channel.provider), data.config)
    if data.secret is not None and data.secret.get_secret_value():
        channel.secret_ciphertext = encrypt_secret(
            data.secret.get_secret_value(),
            purpose=NOTIFICATION_CREDENTIAL_PURPOSE,
        )
    if data.event_types is not None:
        _set_subscriptions(channel, data.event_types)
    channel.updated_at = utc_now()

    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise NotificationChannelConflictError(data.name or channel.name) from error
    return to_public(channel)


def delete_channel(session: Session, channel_id: int) -> None:
    """Delete one notification channel."""
    channel_crud.delete(session, get_channel(session, channel_id))


def list_channels(session: Session) -> list[NotificationChannelPublic]:
    """List safe public notification channel projections."""
    return [to_public(channel) for channel in channel_crud.get_all(session)]


def runtime_channels_for_event(
    session: Session,
    event_type: str,
    target_channel_id: int | None = None,
) -> list[RuntimeNotificationChannel]:
    """Load and decrypt the enabled channels required for one event."""
    if target_channel_id is not None:
        channel = channel_crud.get(session, target_channel_id)
        channels = [channel] if channel is not None and channel.enabled else []
    else:
        channels = list(channel_crud.get_for_event(session, event_type))

    return [
        RuntimeNotificationChannel(
            id=channel.id,
            provider=NotificationProvider(channel.provider),
            config=channel.config,
            secret=decrypt_secret(
                channel.secret_ciphertext,
                purpose=NOTIFICATION_CREDENTIAL_PURPOSE,
            ),
            version=channel.updated_at.isoformat(),
        )
        for channel in channels
    ]
