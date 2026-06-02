import io
from datetime import datetime
from typing import Annotated

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_provider as provider_crud
import app.crud.setting.llm_setting as setting_crud
import app.crud.setting.llm_statistic as statistic_crud
from app.core.config import get_settings
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
from llm.models import invalidate_model_cache

router = APIRouter()

DB_DSN = get_settings().persistent.database.DB_URL


# ==================== LLM Provider Endpoints ====================
@router.post('/providers')
async def create_llm_provider(
    provider_data: LLMProviderCreate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Create a new LLM provider."""
    provider = LLMProvider.model_validate(provider_data.model_dump())
    # pyrefly: ignore [bad-return]
    return provider_crud.insert(session, provider)


@router.get('/providers')
async def get_llm_providers(session: SessionDep, _current_user: AdminDep) -> list[LLMProviderPublic]:
    """Get all LLM providers."""
    # pyrefly: ignore [bad-return]
    return provider_crud.get_all(session)


@router.get('/providers/{provider_id}')
async def get_llm_provider(provider_id: int, session: SessionDep, _current_user: AdminDep) -> LLMProviderPublic:
    """Get a single LLM provider by ID."""
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    # pyrefly: ignore [bad-return]
    return provider


@router.patch('/providers/{provider_id}')
async def update_llm_provider(
    provider_id: int, provider_data: LLMProviderUpdate, session: SessionDep, _current_user: AdminDep
) -> LLMProviderPublic:
    """Update an existing LLM provider."""
    provider = provider_crud.update(session, provider_id, provider_data)
    if provider is None:
        raise HTTPException(status_code=404, detail='Provider not found')
    # pyrefly: ignore [bad-return]
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
    model = LLMModel(**model_data.model_dump())
    # pyrefly: ignore [bad-return]
    return model_crud.insert(session, model)


@router.get('/models')
async def get_llm_models(session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all LLM models."""
    results = model_crud.get_all(session)
    return [LLMModelPublic.model_validate({'provider_name': r.provider.name, **r.model_dump()}) for r in results]


@router.get('/models/{model_id}')
async def get_llm_model(model_id: int, session: SessionDep, _current_user: AdminDep) -> LLMModelPublic:
    """Get a single LLM model by ID."""
    # pyrefly: ignore [bad-return]
    return model_crud.get(session, model_id)


@router.get('/providers/{provider_id}/models')
async def get_provider_models(provider_id: int, session: SessionDep, _current_user: AdminDep) -> list[LLMModelPublic]:
    """Get all models belonging to a specific provider."""
    # pyrefly: ignore [bad-return]
    return model_crud.get_by_provider(session, provider_id)


@router.patch('/models/{model_id}')
async def update_llm_model(
    model_id: int,
    model_data: LLMModelUpdate,
    session: SessionDep,
    _current_user: AdminDep,
) -> LLMModelPublic:
    """Update an existing LLM model."""
    # pyrefly: ignore [bad-return]
    return model_crud.update(session, model_id, model_data)


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
    # pyrefly: ignore [bad-return]
    return setting_crud.get_all_with_details(session)


@router.put('/settings')
async def update_llm_setting(setting_data: LLMSettingUpdate, session: SessionDep, _current_user: AdminDep) -> dict:
    """Update an LLM setting.

    Updates the setting for the specified key, associating it with a new model_id.

    - **key**: setting key (e.g. chat_model, agent_model)
    - **model_id**: target model ID to associate
    """
    setting_crud.update(session, setting_data.key, setting_data.model_id)
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

    Returns a list of detail records, including model name (joined from model_id).

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
        .join(model_df.select(['id', 'name']), left_on='model_id', right_on='id', how='left')
        .rename({'name': 'llm_name'})
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
