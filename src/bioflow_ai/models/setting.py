from datetime import datetime
from decimal import Decimal
from enum import Enum, StrEnum

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from bioflow_ai.models.base import Base
from bioflow_ai.models.user import User
from bioflow_ai.utils.system import utc_now


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


class LLMProvider(Base):
    __tablename__ = 'llm_providers'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(unique=True)
    provider_type: Mapped[ProviderType] = mapped_column(default=ProviderType.OPENAI)
    base_url: Mapped[str | None] = mapped_column(default=None)
    api_key_ciphertext: Mapped[str | None] = mapped_column('api_key', default=None)
    use_proxy: Mapped[bool] = mapped_column(default=False)
    is_active: Mapped[bool] = mapped_column(default=True)

    models: Mapped[list['LLMModel']] = relationship(back_populates='provider')

    @property
    def credential_configured(self) -> bool:
        """Whether this provider has an API credential stored."""
        return bool(self.api_key_ciphertext)


def is_deepseek_provider(provider: LLMProvider) -> bool:
    """Return whether an OpenAI-compatible provider is a direct DeepSeek endpoint."""
    provider_name = provider.name.casefold()
    base_url = (provider.base_url or '').casefold()
    return provider.provider_type is ProviderType.OPENAI and (
        'deepseek' in provider_name or 'api.deepseek.com' in base_url
    )


class LLMModel(Base):
    __tablename__ = 'llm_models'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    provider_id: Mapped[int | None] = mapped_column(ForeignKey(LLMProvider.id), default=None)
    display_name: Mapped[str]
    name: Mapped[str]

    input_price: Mapped[Decimal]
    output_price: Mapped[Decimal]
    cache_read_price: Mapped[Decimal]
    context_window: Mapped[int | None] = mapped_column(default=None)

    reasoning_effort: Mapped[str | None] = mapped_column(default=None)
    is_active: Mapped[bool] = mapped_column(default=True)

    provider: Mapped[LLMProvider | None] = relationship(back_populates='models')
    prices: Mapped[list['LLMStatistic']] = relationship(back_populates='model')


class LLMSetting(Base):
    __tablename__ = 'llm_settings'

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    chat_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    vision_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    agent_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    coding_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    simple_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    long_context_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    high_performance_model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)


class LLMStatistic(Base):
    __tablename__ = 'llm_statistics'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_name: Mapped[str]
    model_id: Mapped[int | None] = mapped_column(ForeignKey(LLMModel.id), default=None)
    setting_key: Mapped[str | None] = mapped_column(default=None)
    user_id: Mapped[int | None] = mapped_column(ForeignKey(User.id), default=None, nullable=True)
    thread: Mapped[str | None] = mapped_column(default=None)

    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cache_read: Mapped[int | None] = mapped_column(default=0)
    cache_creation: Mapped[int] = mapped_column(default=0)
    reasoning_tokens: Mapped[int] = mapped_column(default=0)
    price: Mapped[Decimal]

    time: Mapped[datetime | None] = mapped_column(default=utc_now)

    model: Mapped[LLMModel | None] = relationship(back_populates='prices')
    user: Mapped[User | None] = relationship(User)
