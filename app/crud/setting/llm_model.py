from sqlmodel import Session, col, select

from app.models.setting import LLMModel, LLMProvider
from app.schemas.setting import LLMModelUpdate


def insert(session: Session, model: LLMModel) -> LLMModel:
    """Insert a new LLM model into the database."""
    session.add(model)
    session.commit()
    session.refresh(model)

    return model


def get(session: Session, model_id: int) -> LLMModel:
    """Retrieve an LLM model by its ID."""
    statement = select(LLMModel).where(LLMModel.id == model_id)

    return session.exec(statement).one()


def get_all(session: Session) -> list[LLMModel]:
    """Get all LLM models from active providers."""
    statement = select(LLMModel).join(LLMProvider).where(LLMProvider.is_active).order_by(col(LLMModel.id).asc())

    return session.exec(statement).all()


def get_by_provider(session: Session, provider_id: int) -> list[LLMModel]:
    """Get all LLM models for a given provider."""
    statement = select(LLMModel).where(LLMModel.provider_id == provider_id).order_by(col(LLMModel.id).desc())

    return session.exec(statement).all()


def update(session: Session, model_id: int, model_update: LLMModelUpdate) -> LLMModel:
    """Update an existing LLM model in the database."""
    model = get(session, model_id)
    update_data = model_update.model_dump(exclude_unset=True)
    model.sqlmodel_update(update_data)

    session.add(model)
    session.commit()
    session.refresh(model)

    return model


def delete(session: Session, model_id: int) -> bool:
    """Delete an LLM model by ID, returning True if successful."""
    model = get(session, model_id)
    if model is None:
        return False

    session.delete(model)
    session.commit()

    return True
