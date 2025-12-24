from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User


class ProviderType(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GOOGLE = 'google'


class LLMProvider(SQLModel, table=True):
    __tablename__ = 'llm_providers'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    provider_type: ProviderType = Field(default=ProviderType.OPENAI)
    base_url: Optional[str] = Field(default=None)
    api_key: Optional[str] = Field(default=None)
    use_proxy: bool = Field(default=False)
    is_active: bool = Field(default=True)

    models: list['LLMModel'] = Relationship(back_populates='provider')


class LLMModel(SQLModel, table=True):
    __tablename__ = 'llm_models'

    id: Optional[int] = Field(default=None, primary_key=True)
    provider_id: Optional[int] = Field(default=None, foreign_key='llm_providers.id')
    name: str

    input_price: Decimal
    output_price: Decimal
    cache_read_price: Decimal

    extra_body: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    is_active: bool = Field(default=True)

    provider: Optional['LLMProvider'] = Relationship(back_populates='models')
    prices: list['LLMStatistic'] = Relationship(back_populates='model')


class LLMSetting(SQLModel, table=True):
    __tablename__ = 'llm_settings'

    key: str = Field(primary_key=True)
    model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')

    model: Optional['LLMModel'] = Relationship()


class LLMStatistic(SQLModel, table=True):
    __tablename__ = 'llm_statistics'

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_name: str
    model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    setting_key: Optional[str] = Field(default=None, foreign_key='llm_settings.key')
    user_id: Optional[int] = Field(default=None, foreign_key='users.id')

    input_tokens: int
    output_tokens: int
    cache_read: Optional[int] = Field(default=0)
    price: Decimal

    time: Optional[datetime] = Field(default_factory=lambda: datetime.now(UTC))

    model: Optional['LLMModel'] = Relationship()
    user: Optional['User'] = Relationship()
