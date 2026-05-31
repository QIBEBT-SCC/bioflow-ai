from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from app.models.setting import LLMModel, ModelRole, ProviderType


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
    name: str
    input_price: float
    output_price: float
    cache_read_price: float
    extra_body: dict[str, Any] = {}
    is_active: bool = True


class LLMModelUpdate(BaseModel):
    name: str | None = None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None
    extra_body: dict[str, Any] | None = None
    is_active: bool | None = None


class LLMModelPublic(BaseModel):
    id: int
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
    model_name: str
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
    model_name: str


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
    model_name: str | None = None
    setting_key: str | None = None
    input_tokens: int
    output_tokens: int
    cache_read: int
    price: Decimal
    time: datetime


class LLMStatisticDetailsResponse(BaseModel):
    total: int
    items: list[LLMStatisticDetail]
