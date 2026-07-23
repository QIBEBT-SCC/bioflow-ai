from datetime import datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel

from app.models.user import User  # noqa: TC001
from app.utils.system import utc_now


class ProviderType(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GOOGLE = 'google'


class ReasoningEffort(StrEnum):
    NONE = 'none'
    MINIMAL = 'minimal'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    XHIGH = 'xhigh'
    MAX = 'max'


SUPPORTED_REASONING_EFFORTS: dict[ProviderType, frozenset[ReasoningEffort]] = {
    ProviderType.OPENAI: frozenset(ReasoningEffort),
    ProviderType.ANTHROPIC: frozenset(
        {
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
            ReasoningEffort.MAX,
        }
    ),
    ProviderType.GOOGLE: frozenset(
        {
            ReasoningEffort.MINIMAL,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        }
    ),
}
DEEPSEEK_REASONING_EFFORTS = frozenset(
    {
        ReasoningEffort.NONE,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
        ReasoningEffort.MAX,
    }
)


class ModelRole(StrEnum):
    CHAT = 'chat_model'
    VISION = 'vision_model'
    AGENT = 'agent_model'
    CODING = 'coding_model'
    SIMPLE = 'simple_model'
    LONG_CONTEXT = 'long_context_model'
    HIGH_PERFORMANCE = 'high_performance_model'


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


def is_deepseek_provider(provider: LLMProvider) -> bool:
    """Return whether an OpenAI-compatible provider is a direct DeepSeek endpoint."""
    provider_name = provider.name.casefold()
    base_url = (provider.base_url or '').casefold()
    return provider.provider_type is ProviderType.OPENAI and (
        'deepseek' in provider_name or 'api.deepseek.com' in base_url
    )


class LLMModel(SQLModel, table=True):
    __tablename__ = 'llm_models'

    id: Optional[int] = Field(default=None, primary_key=True)
    provider_id: Optional[int] = Field(default=None, foreign_key='llm_providers.id')
    display_name: str
    name: str

    input_price: Decimal
    output_price: Decimal
    cache_read_price: Decimal

    reasoning_effort: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)

    provider: 'LLMProvider' = Relationship(back_populates='models')
    prices: list['LLMStatistic'] = Relationship(back_populates='model')


class LLMSetting(SQLModel, table=True):
    __tablename__ = 'llm_settings'

    id: int = Field(default=1, primary_key=True)
    chat_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    vision_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    agent_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    coding_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    simple_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    long_context_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    high_performance_model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')


class LLMStatistic(SQLModel, table=True):
    __tablename__ = 'llm_statistics'

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_name: str
    model_id: Optional[int] = Field(default=None, foreign_key='llm_models.id')
    setting_key: Optional[str] = Field(default=None)
    user_id: Optional[int] = Field(default=None, foreign_key='users.id')
    thread: Optional[str] = Field(default=None)

    input_tokens: int
    output_tokens: int
    cache_read: Optional[int] = Field(default=0)
    price: Decimal

    time: Optional[datetime] = Field(default_factory=utc_now)

    model: Optional['LLMModel'] = Relationship()
    user: Optional['User'] = Relationship()
