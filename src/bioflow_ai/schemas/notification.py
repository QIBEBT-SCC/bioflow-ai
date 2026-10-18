from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class NotificationProvider(StrEnum):
    WECOM = 'wecom'
    DISCORD = 'discord'
    TELEGRAM = 'telegram'


class NotificationEventType(StrEnum):
    TOOL_CREATE_SUCCEEDED = 'tool.create.succeeded'
    TOOL_CREATE_INTERRUPTED = 'tool.create.interrupted'
    TOOL_CREATE_FAILED = 'tool.create.failed'
    WORKFLOW_BUILD_SUCCEEDED = 'workflow.build.succeeded'
    WORKFLOW_BUILD_INTERRUPTED = 'workflow.build.interrupted'
    WORKFLOW_BUILD_FAILED = 'workflow.build.failed'


NOTIFICATION_TEST_EVENT = 'notification.test'


class NotificationEventDefinition(BaseModel):
    key: NotificationEventType
    name: dict[str, str]
    description: dict[str, str]


class WeComChannelConfig(BaseModel):
    bot_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    ws_url: str = 'wss://openws.work.weixin.qq.com'


class NotificationChannelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    provider: NotificationProvider
    enabled: bool = True
    config: dict[str, Any]
    secret: SecretStr
    event_types: list[NotificationEventType] = Field(default_factory=list)


class NotificationChannelUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    enabled: bool | None = None
    config: dict[str, Any] | None = None
    secret: SecretStr | None = None
    event_types: list[NotificationEventType] | None = None


class NotificationChannelPublic(BaseModel):
    id: int
    name: str
    provider: NotificationProvider
    enabled: bool
    config: dict[str, Any]
    credential_configured: bool
    event_types: list[NotificationEventType]
    last_attempt_at: datetime | None
    last_success_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


class NotificationEvent(BaseModel):
    id: str
    type: str
    source: str
    occurred_at: datetime
    message: str
    target_channel_id: int | None = None


class RuntimeNotificationChannel(BaseModel):
    id: int
    provider: NotificationProvider
    config: dict[str, Any]
    secret: str
    version: str
