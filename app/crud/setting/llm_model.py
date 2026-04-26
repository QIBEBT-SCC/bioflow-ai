from sqlmodel import Session, col, select

from app.models.setting import LLMModel, LLMProvider
from app.schemas.setting import LLMModelUpdate


def insert(session: Session, model: LLMModel) -> LLMModel:
    session.add(model)
    session.commit()
    session.refresh(model)

    return model


def get(session: Session, model_id: int) -> LLMModel:
    statement = select(LLMModel).where(LLMModel.id == model_id)

    return session.exec(statement).one()


def get_all(session: Session) -> list[LLMModel]:
    statement = select(LLMModel).join(LLMProvider).where(LLMProvider.is_active).order_by(col(LLMModel.id).asc())

    return session.exec(statement).all()


def get_by_provider(session: Session, provider_id: int):
    statement = select(LLMModel).where(LLMModel.provider_id == provider_id).order_by(col(LLMModel.id).desc())

    return session.exec(statement).all()


def update(session: Session, model_id: int, model_update: LLMModelUpdate) -> LLMModel:
    model = get(session, model_id)
    update_data = model_update.model_dump(exclude_unset=True)
    model.sqlmodel_update(update_data)

    session.add(model)
    session.commit()
    session.refresh(model)

    return model


def delete(session: Session, model_id: int) -> bool:
    model = get(session, model_id)
    if model is None:
        return False

    session.delete(model)
    session.commit()

    return True
