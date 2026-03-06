from datetime import datetime
from typing import Annotated, Optional

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
@router.post('/providers')
async def create_llm_provider(
    provider_data: LLMProviderCreate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Create a new LLM provider."""
    provider = LLMProvider.model_validate(provider_data.model_dump())
    return provider_crud.insert(session, provider)


@router.get('/providers')
async def get_llm_providers(session: SessionDep, _current_user: AdminDep) -> list[LLMProviderPublic]:
    """Get all LLM providers."""
    return provider_crud.get_all(session)


@router.get('/providers/{provider_id}')
async def get_llm_provider(provider_id: int, session: SessionDep, _current_user: AdminDep) -> LLMProviderPublic:
    """Get a single LLM provider by ID."""
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    return provider


@router.patch('/providers/{provider_id}')
async def update_llm_provider(
    provider_id: int, provider_data: LLMProviderUpdate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Update an existing LLM provider."""
    provider = provider_crud.update(session, provider_id, provider_data)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    return provider


@router.delete('/providers/{provider_id}')
async def delete_llm_provider(provider_id: int, session: SessionDep, _current_user: AdminDep) -> dict:
    """Delete an LLM provider by ID."""
    success = provider_crud.delete(session, provider_id)
    if not success:
        raise HTTPException(status_code=404, detail='Provider not found')
    return {'message': 'Provider deleted successfully'}


# ==================== LLM Model Endpoints ====================
@router.post('/models')
async def create_llm_model(model_data: LLMModelCreate, session: SessionDep, _current_user: AdminDep) -> LLMModelPublic:
    """Create a new LLM model under an existing provider."""
    provider = provider_crud.get(session, model_data.provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')

    model = LLMModel(**model_data.model_dump())
    return model_crud.insert(session, model)


@router.get('/models')
async def get_llm_models(session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all LLM models."""
    return model_crud.get_all(session)


@router.get('/models/{model_id}')
async def get_llm_model(model_id: int, session: SessionDep, _current_user: AdminDep) -> LLMModelPublic:
    """Get a single LLM model by ID."""
    model = model_crud.get(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')
    return model


@router.get('/providers/{provider_id}/models')
async def get_provider_models(provider_id: int, session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all models belonging to a specific provider."""
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    return model_crud.get_by_provider(session, provider_id)


@router.patch('/models/{model_id}')
async def update_llm_model(
    model_id: int, model_data: LLMModelUpdate, session: SessionDep, _current_user: AdminDep
) -> LLMModelPublic:
    """Update an existing LLM model."""
    model = model_crud.update(session, model_id, model_data)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')
    return model


@router.delete('/models/{model_id}')
async def delete_llm_model(model_id: int, session: SessionDep, _current_user: AdminDep) -> dict:
    """Delete an LLM model by ID."""
    success = model_crud.delete(session, model_id)
    if not success:
        raise HTTPException(status_code=404, detail='Model not found')
    return {'message': 'Model deleted successfully'}


# ==================== LLM Setting Endpoints ====================
@router.get('/settings')
async def get_llm_settings(session: SessionDep, _current_user: AdminDep) -> LLMSettingPublic:
    """Get all LLM settings.

    Returns all settings, each containing model_id, model_name and provider_name.
    """
    return setting_crud.get_all_with_details(session)


@router.put('/settings')
async def update_llm_setting(setting_data: LLMSettingUpdate, session: SessionDep, _current_user: AdminDep) -> dict:
    """Update an LLM setting.

    Updates the setting for the specified key, associating it with a new model_id.

    - **key**: setting key (e.g. chat_model, agent_model)
    - **model_id**: target model ID to associate
    """
    model = model_crud.get(session, setting_data.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail='Model not found')

    setting_crud.update(session, setting_data.key, setting_data.model_id)
    return {'message': f'Setting {setting_data.key} updated successfully'}


# ==================== LLM Statistic Endpoints ====================
@router.get('/statistics/overview')
async def get_llm_statistics_overview(
    session: SessionDep,
    _current_user: AdminDep,
    start_date: Annotated[
        Optional[datetime], Query(description='Start date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    end_date: Annotated[
        Optional[datetime], Query(description='End date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
) -> LLMStatisticOverview:
    """Get LLM usage statistics overview.

    Returns overall statistics and grouped statistics by Agent, Model, Type and User.

    - **start_date**: optional start time; no lower bound if not provided
    - **end_date**: optional end time; no upper bound if not provided
    """
    return statistic_crud.get_summary(session, start_date, end_date)


@router.get('/statistics/details')
async def get_llm_statistics_details(
    session: SessionDep,
    _current_user: AdminDep,
    start_date: Annotated[
        Optional[datetime], Query(description='Start date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    end_date: Annotated[
        Optional[datetime], Query(description='End date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description='Page size')] = 100,
    offset: Annotated[int, Query(ge=0, description='Offset for pagination')] = 0,
) -> LLMStatisticDetailsResponse:
    """Get detailed LLM usage records.

    Returns a list of detail records, including model name (joined from model_id).

    - **start_date**: optional start time; no lower bound if not provided
    - **end_date**: optional end time; no upper bound if not provided
    - **limit**: page size, default 100, max 1000
    - **offset**: pagination offset
    """
    return statistic_crud.get_details(session, start_date, end_date, limit, offset)
