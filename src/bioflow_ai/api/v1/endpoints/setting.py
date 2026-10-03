import io
from datetime import datetime
from typing import Annotated

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

import bioflow_ai.crud.setting.llm_model as model_crud
import bioflow_ai.crud.setting.llm_provider as provider_crud
import bioflow_ai.crud.setting.llm_setting as setting_crud
import bioflow_ai.crud.setting.llm_statistic as statistic_crud
from bioflow_ai.core.auth import AdminDep
from bioflow_ai.core.config import get_settings
from bioflow_ai.infrastructure.database import SessionDep
from bioflow_ai.llm.models import invalidate_model_cache
from bioflow_ai.models.setting import (
    DEEPSEEK_REASONING_EFFORTS,
    SUPPORTED_REASONING_EFFORTS,
    LLMModel,
    LLMProvider,
    ReasoningEffort,
    is_deepseek_provider,
)
from bioflow_ai.schemas.setting import (
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
from bioflow_ai.services import llm_setting as setting_service

router = APIRouter()

DB_DSN = get_settings().persistent.database.DB_URL


def _validate_reasoning_effort(provider: LLMProvider, reasoning_effort: ReasoningEffort | None) -> None:
    supported_efforts = (
        DEEPSEEK_REASONING_EFFORTS
        if is_deepseek_provider(provider)
        else SUPPORTED_REASONING_EFFORTS[provider.provider_type]
    )
    if reasoning_effort is None or reasoning_effort in supported_efforts:
        return

    supported = ', '.join(sorted(effort.value for effort in supported_efforts))
    raise HTTPException(
        status_code=422,
        detail=f'Reasoning effort {reasoning_effort.value!r} is not supported by {provider.name}; use: {supported}',
    )


def _get_provider(session: SessionDep, provider_id: int) -> LLMProvider:
    try:
        return setting_service.get_provider(session, provider_id)
    except setting_service.LLMProviderNotFoundError as error:
        raise HTTPException(status_code=404, detail='Provider not found') from error


def _get_model(session: SessionDep, model_id: int) -> LLMModel:
    try:
        return setting_service.get_model(session, model_id)
    except setting_service.LLMModelNotFoundError as error:
        raise HTTPException(status_code=404, detail='Model not found') from error


def _to_model_public(model: LLMModel, *, include_provider_name: bool = False) -> LLMModelPublic:
    projection = LLMModelPublic.model_validate(model)
    if include_provider_name:
        return projection.model_copy(
            update={'provider_name': model.provider.name if model.provider is not None else None}
        )
    return projection


def _to_provider_public(provider: LLMProvider) -> LLMProviderPublic:
    return LLMProviderPublic.model_validate(provider)


# ==================== LLM Provider Endpoints ====================
@router.post('/providers')
async def create_llm_provider(
    provider_data: LLMProviderCreate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Create a new LLM provider."""
    return _to_provider_public(setting_service.create_provider(session, provider_data))


@router.get('/providers')
async def get_llm_providers(session: SessionDep, _current_user: AdminDep) -> list[LLMProviderPublic]:
    """Get all LLM providers."""
    return [_to_provider_public(provider) for provider in provider_crud.get_all(session)]


@router.get('/providers/{provider_id}')
async def get_llm_provider(provider_id: int, session: SessionDep, _current_user: AdminDep) -> LLMProviderPublic:
    """Get a single LLM provider by ID."""
    return _to_provider_public(_get_provider(session, provider_id))


@router.patch('/providers/{provider_id}')
async def update_llm_provider(
    provider_id: int, provider_data: LLMProviderUpdate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Update an existing LLM provider."""
    try:
        provider = setting_service.update_provider(session, provider_id, provider_data)
    except setting_service.LLMProviderNotFoundError as error:
        raise HTTPException(status_code=404, detail='Provider not found') from error
    invalidate_model_cache()
    return _to_provider_public(provider)


@router.delete('/providers/{provider_id}')
async def delete_llm_provider(provider_id: int, session: SessionDep, _current_user: AdminDep) -> dict:
    """Delete an LLM provider by ID."""
    try:
        setting_service.delete_provider(session, provider_id)
    except setting_service.LLMProviderNotFoundError as error:
        raise HTTPException(status_code=404, detail='Provider not found') from error
    invalidate_model_cache()
    return {'message': 'Provider deleted successfully'}


# ==================== LLM Model Endpoints ====================
@router.post('/models')
async def create_llm_model(model_data: LLMModelCreate, session: SessionDep, _current_user: AdminDep) -> LLMModelPublic:
    """Create a new LLM model under an existing provider."""
    provider = _get_provider(session, model_data.provider_id)
    _validate_reasoning_effort(provider, model_data.reasoning_effort)
    model = LLMModel(**model_data.model_dump())
    return _to_model_public(setting_service.create_model(session, model))


@router.get('/models')
async def get_llm_models(session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all LLM models."""
    results = model_crud.get_all(session)
    return [_to_model_public(model, include_provider_name=True) for model in results]


@router.get('/models/{model_id}')
async def get_llm_model(model_id: int, session: SessionDep, _current_user: AdminDep) -> LLMModelPublic:
    """Get a single LLM model by ID."""
    return _to_model_public(_get_model(session, model_id))


@router.get('/providers/{provider_id}/models')
async def get_provider_models(provider_id: int, session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all models belonging to a specific provider."""
    return [_to_model_public(model) for model in model_crud.get_by_provider(session, provider_id)]


@router.patch('/models/{model_id}')
async def update_llm_model(
    model_id: int,
    model_data: LLMModelUpdate,
    session: SessionDep,
    _current_user: AdminDep,
) -> LLMModelPublic:
    """Update an existing LLM model."""
    model = _get_model(session, model_id)
    if model.provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    _validate_reasoning_effort(model.provider, model_data.reasoning_effort)
    try:
        updated_model = setting_service.update_model(session, model_id, model_data)
    except setting_service.LLMModelNotFoundError as error:
        raise HTTPException(status_code=404, detail='Model not found') from error
    invalidate_model_cache()
    return _to_model_public(updated_model)


@router.delete('/models/{model_id}')
async def delete_llm_model(model_id: int, session: SessionDep, _current_user: AdminDep) -> dict:
    """Delete an LLM model by ID."""
    try:
        setting_service.delete_model(session, model_id)
    except setting_service.LLMModelNotFoundError as error:
        raise HTTPException(status_code=404, detail='Model not found') from error
    return {'message': 'Model deleted successfully'}


# ==================== LLM Setting Endpoints ====================
@router.get('/settings')
async def get_llm_settings(session: SessionDep, _current_user: AdminDep) -> LLMSettingPublic:
    """Get all LLM settings.

    Returns all settings, each containing the model ID, display name, API name, and provider name.
    """
    # pyrefly: ignore [bad-return]
    return setting_crud.get_all_with_details(session)


@router.put('/settings')
async def update_llm_setting(setting_data: LLMSettingUpdate, session: SessionDep, _current_user: AdminDep) -> dict:
    """Update an LLM setting.

    Updates the setting for the specified key, associating it with a new model_id.

    - **key**: setting key (e.g. chat_model, agent_model)
    - **model_id**: target model ID to associate
    """
    setting_service.update_role(session, setting_data.key, setting_data.model_id)
    invalidate_model_cache(setting_data.key)
    return {'message': f'Setting {setting_data.key.value} updated successfully'}


# ==================== LLM Statistic Endpoints ====================
@router.get('/statistics/overview')
async def get_llm_statistics_overview(
    session: SessionDep,
    _current_user: AdminDep,
    start_date: Annotated[
        datetime | None, Query(description='Start date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    end_date: Annotated[datetime | None, Query(description='End date (inclusive), format: YYYY-MM-DDTHH:MM:SS')] = None,
) -> LLMStatisticOverview:
    """Get LLM usage statistics overview.

    Returns overall statistics and grouped statistics by Agent, Model, Type and User.

    - **start_date**: optional start time; no lower bound if not provided
    - **end_date**: optional end time; no upper bound if not provided
    """
    # pyrefly: ignore [bad-return]
    return statistic_crud.get_summary(session, start_date, end_date)


@router.get('/statistics/details')
async def get_llm_statistics_details(
    session: SessionDep,
    _current_user: AdminDep,
    start_date: Annotated[
        datetime | None, Query(description='Start date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    end_date: Annotated[datetime | None, Query(description='End date (inclusive), format: YYYY-MM-DDTHH:MM:SS')] = None,
    limit: Annotated[int, Query(ge=1, le=1000, description='Page size')] = 100,
    offset: Annotated[int, Query(ge=0, description='Offset for pagination')] = 0,
) -> LLMStatisticDetailsResponse:
    """Get detailed LLM usage records.

    Returns a list of detail records, including the model display name and API name joined from model_id.

    - **start_date**: optional start time; no lower bound if not provided
    - **end_date**: optional end time; no upper bound if not provided
    - **limit**: page size, default 100, max 1000
    - **offset**: pagination offset
    """
    # pyrefly: ignore [bad-return]
    return statistic_crud.get_details(session, start_date, end_date, limit, offset)


@router.get('/statistics/details/download')
async def download_statistic_details(
    _current_user: AdminDep,
    start_date: Annotated[
        datetime | None, Query(description='Start date (inclusive), format: YYYY-MM-DDTHH:MM:SS')
    ] = None,
    end_date: Annotated[datetime | None, Query(description='End date (inclusive), format: YYYY-MM-DDTHH:MM:SS')] = None,
) -> StreamingResponse:
    """Download LLM usage records as CSV.

    - **start_date**: optional start time; no lower bound if not provided
    - **end_date**: optional end time; no upper bound if not provided
    """
    model_df = pl.read_database_uri(
        'SELECT * FROM llm_models',
        uri=DB_DSN.unicode_string().replace(DB_DSN.scheme, DB_DSN.scheme.split('+')[0]),
        partition_on='id',
        partition_num=10,
        engine='connectorx',
    ).sort('id')

    statistic_df = (
        pl.read_database_uri(
            'SELECT * FROM llm_statistics',
            uri=DB_DSN.unicode_string().replace(DB_DSN.scheme, DB_DSN.scheme.split('+')[0]),
            partition_on='id',
            partition_num=10,
            engine='connectorx',
        )
        .join(model_df.select(['id', 'display_name', 'name']), left_on='model_id', right_on='id', how='left')
        .sort('id')
    )

    if start_date is not None:
        statistic_df = statistic_df.filter(pl.col('time') >= start_date.replace(tzinfo=None))
    if end_date is not None:
        statistic_df = statistic_df.filter(pl.col('time') <= end_date.replace(tzinfo=None))

    buf = io.BytesIO()
    statistic_df.write_csv(buf)
    buf.seek(0)

    filename = 'llm_statistics'
    if start_date:
        filename += f'_from_{start_date.strftime("%Y%m%d")}'
    if end_date:
        filename += f'_to_{end_date.strftime("%Y%m%d")}'
    filename += '.csv'

    return StreamingResponse(
        buf,
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )
