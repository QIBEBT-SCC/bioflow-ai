from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.setting import LLMModel, ModelRole, ProviderType
from app.utils.model_pricing import find_model_price


# ==================== LLMProvider Schemas ====================
class LLMProviderCreate(BaseModel):
    name: str
    provider_type: ProviderType
    base_url: str | None = None
    api_key: str
    use_proxy: bool = False
    is_active: bool = True


class LLMProviderUpdate(BaseModel):
    name: str | None = None
    provider_type: ProviderType | None = None
    base_url: str | None = None
    api_key: str | None = None
    use_proxy: bool | None = None
    is_active: bool | None = None


class LLMProviderPublic(BaseModel):
    id: int
    name: str
    provider_type: ProviderType
    base_url: str | None = None
    api_key: str | None = None
    use_proxy: bool
    is_active: bool
    models: list[LLMModel]


# ==================== LLMModel Schemas ====================
class LLMModelCreate(BaseModel):
    provider_id: int
    display_name: str = Field(min_length=1)
    name: str
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

    @model_validator(mode='after')
    def populate_default_prices(self) -> 'LLMModelCreate':
        """Fill omitted prices from the model catalog while preserving user overrides."""
        default_price = find_model_price(self.name)
        missing_fields = [
            field_name
            for field_name in ('input_price', 'output_price', 'cache_read_price')
            if getattr(self, field_name) is None
        ]
        if default_price is None and missing_fields:
            fields = ', '.join(missing_fields)
            message = f'No default price found for model {self.name!r}; provide: {fields}'
            raise ValueError(message)

        if default_price is not None:
            if self.input_price is None:
                self.input_price = float(default_price.input)
            if self.output_price is None:
                self.output_price = float(default_price.output)
            if self.cache_read_price is None:
                self.cache_read_price = float(default_price.cache_read)
        return self


class LLMModelUpdate(BaseModel):
    display_name: str | None = Field(default=None, min_length=1)
    name: str | None = None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None
    extra_body: dict[str, Any] | None = None
    is_active: bool | None = None


class LLMModelPublic(BaseModel):
    id: int
    display_name: str
    name: str
    input_price: float
    output_price: float
    cache_read_price: float
    extra_body: dict[str, Any]
    is_active: bool

    provider_name: str | None = None


# ==================== LLMSetting Schemas ====================
class LLMSettingItem(BaseModel):
    model_id: int
    display_name: str
    name: str
    provider_name: str


class LLMSettingPublic(BaseModel):
    chat_model: LLMSettingItem
    vision_model: LLMSettingItem
    agent_model: LLMSettingItem
    coding_model: LLMSettingItem
    simple_model: LLMSettingItem
    long_context_model: LLMSettingItem
    high_performance_model: LLMSettingItem


class LLMSettingUpdate(BaseModel):
    key: ModelRole
    model_id: int


# ==================== LLMStatistic Schemas ====================
class LLMStatisticSummary(BaseModel):
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read: int
    total_price: Decimal


class LLMStatisticByAgent(LLMStatisticSummary):
    count: int
    agent_name: str


class LLMStatisticByModel(LLMStatisticSummary):
    count: int
    display_name: str
    name: str


class LLMStatisticByType(LLMStatisticSummary):
    count: int
    setting_key: str


class LLMStatisticByUser(LLMStatisticSummary):
    count: int
    user_name: str


class LLMStatisticOverview(BaseModel):
    total: LLMStatisticSummary
    by_agent: list[LLMStatisticByAgent]
    by_model: list[LLMStatisticByModel]
    by_type: list[LLMStatisticByType]
    by_user: list[LLMStatisticByUser]


class LLMStatisticDetail(BaseModel):
    id: int
    agent_name: str
    display_name: str | None = None
    name: str | None = None
    setting_key: str | None = None
    input_tokens: int
    output_tokens: int
    cache_read: int
    price: Decimal
    time: datetime


class LLMStatisticDetailsResponse(BaseModel):
    total: int
    items: list[LLMStatisticDetail]
