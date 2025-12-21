from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


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

    input_price: float
    output_price: float
    cache_read_price: float

    extra_body: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    is_active: bool = Field(default=True)

    provider: Optional['LLMProvider'] = Relationship(back_populates='models')
