from datetime import datetime

from sqlalchemy import JSON, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from bioflow_ai.models.base import Base
from bioflow_ai.utils.system import utc_now


class NotificationChannel(Base):
    """One configured outbound notification destination."""

    __tablename__ = 'notification_channels'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    provider: Mapped[str] = mapped_column(String(32))
    enabled: Mapped[bool] = mapped_column(default=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    secret_ciphertext: Mapped[str] = mapped_column(Text)
    last_attempt_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_success_at: Mapped[datetime | None] = mapped_column(nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now)

    subscriptions: Mapped[list['NotificationSubscription']] = relationship(
        back_populates='channel',
        cascade='all, delete-orphan',
    )


class NotificationSubscription(Base):
    """Route one notification event type to a configured channel."""

    __tablename__ = 'notification_subscriptions'
    __table_args__ = (UniqueConstraint('channel_id', 'event_type'),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey(NotificationChannel.id, ondelete='CASCADE'))
    event_type: Mapped[str] = mapped_column(String(120))

    channel: Mapped[NotificationChannel] = relationship(back_populates='subscriptions')
