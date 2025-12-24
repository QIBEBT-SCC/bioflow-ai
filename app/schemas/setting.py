from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.models.setting import LLMModel, ProviderType


# ==================== LLMProvider Schemas ====================
class LLMProviderCreate(BaseModel):
    name: str
    provider_type: ProviderType
    base_url: Optional[str] = None
    api_key: str
    use_proxy: bool = False
    is_active: bool = True


class LLMProviderUpdate(BaseModel):
    name: Optional[str] = None
    provider_type: Optional[ProviderType] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_proxy: Optional[bool] = None
    is_active: Optional[bool] = None


class LLMProviderPublic(BaseModel):
    id: int
    name: str
    provider_type: ProviderType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
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
    name: Optional[str] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    cache_read_price: Optional[float] = None
    extra_body: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None


class LLMModelPublic(BaseModel):
    id: int
    provider_id: int
    name: str
    input_price: float
    output_price: float
    cache_read_price: float
    extra_body: dict[str, Any]
    is_active: bool


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
    key: Literal[
        'chat_model',
        'vision_model',
        'agent_model',
        'coding_model',
        'simple_model',
        'long_context_model',
        'high_performance_model',
    ]
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
    """LLM 使用统计概览，包含总体统计和各维度分组"""

    total: LLMStatisticSummary
    by_agent: list[LLMStatisticByAgent]
    by_model: list[LLMStatisticByModel]
    by_type: list[LLMStatisticByType]
    by_user: list[LLMStatisticByUser]


class LLMStatisticDetail(BaseModel):
    id: int
    agent_name: str
    model_name: Optional[str] = None
    setting_key: Optional[str] = None
    input_tokens: int
    output_tokens: int
    cache_read: int
    price: Decimal
    time: datetime


class LLMStatisticDetailsResponse(BaseModel):
    """LLM 使用详细记录响应"""

    total: int
    items: list[LLMStatisticDetail]
