from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models.setting import LLMModel, LLMProvider
from app.schemas.setting import LLMModelUpdate


def insert(session: Session, model: LLMModel) -> LLMModel:
    """Stage a new LLM model."""
    session.add(model)
    session.flush()
    session.refresh(model)
    return model


def get(session: Session, model_id: int) -> LLMModel | None:
    """Return an LLM model by ID, including its provider."""
    statement = select(LLMModel).options(joinedload(LLMModel.provider)).where(LLMModel.id == model_id)
    return session.scalar(statement)


def get_all(session: Session) -> Sequence[LLMModel]:
    """Get all LLM models from active providers."""
    statement = (
        select(LLMModel)
        .options(joinedload(LLMModel.provider))
        .join(LLMProvider)
        .where(LLMProvider.is_active)
        .order_by(LLMModel.id.asc())
    )
    return session.scalars(statement).all()


def get_by_provider(session: Session, provider_id: int) -> Sequence[LLMModel]:
    """Get all LLM models for a given provider."""
    statement = (
        select(LLMModel)
        .options(joinedload(LLMModel.provider))
        .where(LLMModel.provider_id == provider_id)
        .order_by(LLMModel.id.desc())
    )
    return session.scalars(statement).all()


def update(session: Session, model_id: int, model_update: LLMModelUpdate) -> LLMModel | None:
    """Stage updates to an LLM model."""
    model = get(session, model_id)
    if model is None:
        return None
    update_data = model_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)
    session.flush()
    session.refresh(model)
    return model


def delete(session: Session, model_id: int) -> bool:
    """Stage deletion of an LLM model, returning whether it existed."""
    model = get(session, model_id)
    if model is None:
        return False

    session.delete(model)
    session.flush()
    return True
