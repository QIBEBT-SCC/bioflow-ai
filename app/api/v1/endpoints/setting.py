from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_provider as provider_crud
import app.crud.setting.llm_setting as setting_crud
import app.crud.setting.llm_statistic as statistic_crud
from app.core.session import SessionDep
from app.models.setting import LLMModel, LLMProvider
from app.schemas.setting import (
    LLMModelCreate,
    LLMModelPublic,
    LLMModelUpdate,
    LLMProviderCreate,
    LLMProviderPublic,
    LLMProviderUpdate,
    LLMSettingPublic,
    LLMSettingUpdate,
    LLMStatisticDetailsResponse,
    LLMStatisticOverview,
)
from app.utils.security import AdminDep

router = APIRouter()


# ==================== LLM Provider Endpoints ====================
@router.post('/providers', response_model=LLMProviderPublic)
async def create_llm_provider(provider_data: LLMProviderCreate, session: SessionDep, current_user: AdminDep):
    provider = LLMProvider.model_validate(provider_data.model_dump())
    result = provider_crud.insert(session, provider)
    return result


@router.get('/providers', response_model=List[LLMProviderPublic])
async def get_llm_providers(session: SessionDep, current_user: AdminDep):
    return provider_crud.get_all(session)


@router.get('/providers/{provider_id}', response_model=LLMProviderPublic)
async def get_llm_provider(provider_id: int, session: SessionDep, current_user: AdminDep):
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    return provider


@router.patch('/providers/{provider_id}', response_model=LLMProviderPublic)
async def update_llm_provider(
    provider_id: int, provider_data: LLMProviderUpdate, session: SessionDep, current_user: AdminDep
):
    provider = provider_crud.update(session, provider_id, provider_data)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    return provider


@router.delete('/providers/{provider_id}')
async def delete_llm_provider(provider_id: int, session: SessionDep, current_user: AdminDep):
    success = provider_crud.delete(session, provider_id)
    if not success:
        raise HTTPException(status_code=404, detail='Provider not found')
    return {'message': 'Provider deleted successfully'}


# ==================== LLM Model Endpoints ====================
@router.post('/models', response_model=LLMModelPublic)
async def create_llm_model(model_data: LLMModelCreate, session: SessionDep, current_user: AdminDep):
    # 验证 provider 是否存在
    provider = provider_crud.get(session, model_data.provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')

    model = LLMModel(**model_data.model_dump())
    result = model_crud.insert(session, model)
    return result


@router.get('/models', response_model=List[LLMModelPublic])
async def get_llm_models(session: SessionDep, current_user: AdminDep):
    return model_crud.get_all(session)


@router.get('/models/{model_id}', response_model=LLMModelPublic)
async def get_llm_model(model_id: int, session: SessionDep, current_user: AdminDep):
    model = model_crud.get(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')
    return model


@router.get('/providers/{provider_id}/models', response_model=List[LLMModelPublic])
async def get_provider_models(provider_id: int, session: SessionDep, current_user: AdminDep):
    # 验证 provider 是否存在
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')

    return model_crud.get_by_provider(session, provider_id)


@router.patch('/models/{model_id}', response_model=LLMModelPublic)
async def update_llm_model(model_id: int, model_data: LLMModelUpdate, session: SessionDep, current_user: AdminDep):
    model = model_crud.update(session, model_id, model_data)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')
    return model


@router.delete('/models/{model_id}')
async def delete_llm_model(model_id: int, session: SessionDep, current_user: AdminDep):
    success = model_crud.delete(session, model_id)
    if not success:
        raise HTTPException(status_code=404, detail='Model not found')
    return {'message': 'Model deleted successfully'}


# ==================== LLM Setting Endpoints ====================
@router.get('/settings', response_model=LLMSettingPublic)
async def get_llm_settings(session: SessionDep, current_user: AdminDep):
    """
    获取所有 LLM 设置

    返回所有设置项，每个设置包含 model_id、model_name 和 provider_name
    """
    settings = setting_crud.get_all_with_details(session)
    return settings


@router.put('/settings', response_model=dict)
async def update_llm_setting(setting_data: LLMSettingUpdate, session: SessionDep, current_user: AdminDep):
    """
    更新 LLM 设置

    更新指定 key 的设置，将其关联到新的 model_id

    - **key**: 设置的 key（如 chat_model, agent_model 等）
    - **model_id**: 要关联的模型 ID
    """
    # 验证 model 是否存在
    model = model_crud.get(session, setting_data.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')

    setting_crud.update(session, setting_data.key, setting_data.model_id)
    return {'message': f'Setting {setting_data.key} updated successfully'}


# ==================== LLM Statistic Endpoints ====================
@router.get('/statistics/overview', response_model=LLMStatisticOverview)
async def get_llm_statistics_overview(
    session: SessionDep,
    current_user: AdminDep,
    start_date: Optional[datetime] = Query(None, description='开始日期（包含），格式：YYYY-MM-DDTHH:MM:SS'),
    end_date: Optional[datetime] = Query(None, description='结束日期（包含），格式：YYYY-MM-DDTHH:MM:SS'),
):
    """
    获取 LLM 使用统计概览

    包含总体统计和按不同维度（Agent、Model、Type、User）的分组统计

    - **start_date**: 可选，查询起始时间，如果不提供则不限制开始时间
    - **end_date**: 可选，查询结束时间，如果不提供则不限制结束时间
    """
    summary = statistic_crud.get_summary(session, start_date, end_date)
    return summary


@router.get('/statistics/details', response_model=LLMStatisticDetailsResponse)
async def get_llm_statistics_details(
    session: SessionDep,
    current_user: AdminDep,
    start_date: Optional[datetime] = Query(None, description='开始日期（包含），格式：YYYY-MM-DDTHH:MM:SS'),
    end_date: Optional[datetime] = Query(None, description='结束日期（包含），格式：YYYY-MM-DDTHH:MM:SS'),
    limit: int = Query(100, ge=1, le=1000, description='每页记录数'),
    offset: int = Query(0, ge=0, description='偏移量'),
):
    """
    获取 LLM 使用详细记录列表

    返回详细记录列表，包含 model name（从 model_id JOIN 获取）

    - **start_date**: 可选，查询起始时间，如果不提供则不限制开始时间
    - **end_date**: 可选，查询结束时间，如果不提供则不限制结束时间
    - **limit**: 每页记录数，默认 100，最大 1000
    - **offset**: 偏移量，用于分页
    """
    result = statistic_crud.get_details(session, start_date, end_date, limit, offset)
    return result
