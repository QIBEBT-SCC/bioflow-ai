from typing import List

from fastapi import APIRouter, HTTPException

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_provider as provider_crud
from app.core.session import SessionDep
from app.models.setting import LLMModel, LLMProvider
from app.schemas.setting import (
    LLMModelCreate,
    LLMModelPublic,
    LLMModelUpdate,
    LLMProviderCreate,
    LLMProviderPublic,
    LLMProviderUpdate,
)
from app.utils.security import AdminDep

router = APIRouter()


# ==================== LLM Provider Endpoints ====================
@router.post('/providers', response_model=LLMProviderPublic)
async def create_llm_provider(provider_data: LLMProviderCreate, session: SessionDep, current_user: AdminDep):
    provider = LLMProvider(**provider_data.model_dump())
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
