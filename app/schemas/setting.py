from typing import Any, Optional

from pydantic import BaseModel

from app.models.setting import LLMModel


# ==================== LLMProvider Schemas ====================
class LLMProviderCreate(BaseModel):
    name: str
    base_url: str
    api_key: str
    use_proxy: bool
    is_active: bool = True


class LLMProviderUpdate(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_proxy: Optional[bool] = None
    is_active: Optional[bool] = None


class LLMProviderPublic(BaseModel):
    id: int
    name: str
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
    cache_write_price: float
    cache_read_price: float
    extra_body: dict[str, Any] = {}
    is_active: bool = True


class LLMModelUpdate(BaseModel):
    name: Optional[str] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    cache_write_price: Optional[float] = None
    cache_read_price: Optional[float] = None
    extra_body: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None


class LLMModelPublic(BaseModel):
    id: int
    provider_id: int
    name: str
    input_price: float
    output_price: float
    cache_write_price: float
    cache_read_price: float
    extra_body: dict[str, Any]
    is_active: bool
