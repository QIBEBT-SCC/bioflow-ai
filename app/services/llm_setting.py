"""Application services for LLM providers, models, role settings, and usage statistics."""

from collections.abc import Sequence

from sqlalchemy.orm import Session

import app.crud.setting.llm_model as model_crud
import app.crud.setting.llm_provider as provider_crud
import app.crud.setting.llm_setting as setting_crud
import app.crud.setting.llm_statistic as statistic_crud
from app.models.setting import LLMModel, LLMProvider, LLMSetting, LLMStatistic, ModelRole
from app.schemas.setting import LLMModelUpdate, LLMProviderUpdate
from app.services._transaction import committing


class LLMSettingServiceError(ValueError):
    """Base error for LLM-setting application operations."""


class LLMProviderNotFoundError(LLMSettingServiceError):
    """Raised when an LLM provider does not exist."""


class LLMModelNotFoundError(LLMSettingServiceError):
    """Raised when an LLM model does not exist."""


class LLMRoleNotConfiguredError(LLMSettingServiceError):
    """Raised when no model is configured for an LLM role."""


def get_provider(session: Session, provider_id: int) -> LLMProvider:
    """Return one LLM provider or raise a domain error."""
    provider = provider_crud.get(session, provider_id)
    if provider is None:
        raise LLMProviderNotFoundError
    return provider


def create_provider(session: Session, provider: LLMProvider) -> LLMProvider:
    """Create and commit an LLM provider."""
    with committing(session):
        provider_crud.insert(session, provider)
    session.refresh(provider)
    return provider


def update_provider(session: Session, provider_id: int, data: LLMProviderUpdate) -> LLMProvider:
    """Update and commit an LLM provider."""
    with committing(session):
        provider = provider_crud.update(session, provider_id, data)
        if provider is None:
            raise LLMProviderNotFoundError
    session.refresh(provider)
    return provider


def delete_provider(session: Session, provider_id: int) -> None:
    """Delete and commit an LLM provider."""
    with committing(session):
        if not provider_crud.delete(session, provider_id):
            raise LLMProviderNotFoundError


def save_providers(session: Session, providers: Sequence[LLMProvider]) -> None:
    """Commit a batch of provider configuration changes."""
    with committing(session):
        provider_crud.update_all(session, providers)


def get_model(session: Session, model_id: int) -> LLMModel:
    """Return one LLM model or raise a domain error."""
    model = model_crud.get(session, model_id)
    if model is None:
        raise LLMModelNotFoundError
    return model


def create_model(session: Session, model: LLMModel) -> LLMModel:
    """Create and commit an LLM model."""
    with committing(session):
        model_crud.insert(session, model)
    session.refresh(model)
    return model


def update_model(session: Session, model_id: int, data: LLMModelUpdate) -> LLMModel:
    """Update and commit an LLM model."""
    with committing(session):
        model = model_crud.update(session, model_id, data)
        if model is None:
            raise LLMModelNotFoundError
    session.refresh(model)
    return model


def delete_model(session: Session, model_id: int) -> None:
    """Delete and commit an LLM model."""
    with committing(session):
        if not model_crud.delete(session, model_id):
            raise LLMModelNotFoundError


def get_role_model_id(session: Session, role: ModelRole) -> int:
    """Return the configured model ID for one role."""
    model_id = setting_crud.get_role(session, role)
    if model_id is None:
        raise LLMRoleNotConfiguredError
    return model_id


def update_role(session: Session, role: ModelRole, model_id: int) -> LLMSetting:
    """Update and commit one role-to-model assignment."""
    with committing(session):
        setting = setting_crud.update(session, role, model_id)
    session.refresh(setting)
    return setting


def record_statistic(session: Session, statistic: LLMStatistic) -> None:
    """Persist one LLM usage statistic."""
    with committing(session):
        statistic_crud.insert(session, statistic)


def refresh_statistic_thread_id(session: Session, thread_id: str, new_id: str) -> None:
    """Replace a statistic thread ID in one transaction."""
    with committing(session):
        statistic_crud.refresh_thread_id(session, thread_id, new_id)
