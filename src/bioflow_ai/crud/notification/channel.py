from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from bioflow_ai.models.notification import NotificationChannel, NotificationSubscription


def get(session: Session, channel_id: int) -> NotificationChannel | None:
    """Return one notification channel with subscriptions."""
    stmt = (
        select(NotificationChannel)
        .options(selectinload(NotificationChannel.subscriptions))
        .where(NotificationChannel.id == channel_id)
    )
    return session.scalar(stmt)


def get_all(session: Session) -> Sequence[NotificationChannel]:
    """Return all notification channels with subscriptions."""
    stmt = (
        select(NotificationChannel)
        .options(selectinload(NotificationChannel.subscriptions))
        .order_by(NotificationChannel.id)
    )
    return session.scalars(stmt).all()


def get_for_event(session: Session, event_type: str) -> Sequence[NotificationChannel]:
    """Return enabled channels subscribed to an event type."""
    stmt = (
        select(NotificationChannel)
        .join(NotificationSubscription)
        .options(selectinload(NotificationChannel.subscriptions))
        .where(
            NotificationChannel.enabled.is_(True),
            NotificationSubscription.event_type == event_type,
        )
        .order_by(NotificationChannel.id)
    )
    return session.scalars(stmt).all()


def insert(session: Session, channel: NotificationChannel) -> NotificationChannel:
    """Persist one notification channel."""
    session.add(channel)
    session.commit()
    session.refresh(channel)
    return channel


def delete(session: Session, channel: NotificationChannel) -> None:
    """Delete one notification channel."""
    session.delete(channel)
    session.commit()


def record_delivery_result(
    session: Session,
    channel_id: int,
    *,
    attempted_at: datetime,
    succeeded_at: datetime | None,
    error: str | None,
) -> None:
    """Persist the latest delivery outcome without changing the channel configuration version."""
    stmt = (
        update(NotificationChannel)
        .where(NotificationChannel.id == channel_id)
        .values(
            last_attempt_at=attempted_at,
            last_success_at=succeeded_at if succeeded_at is not None else NotificationChannel.last_success_at,
            last_error=error,
        )
    )
    session.execute(stmt)
    session.commit()
