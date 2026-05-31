from sqlalchemy import select
from sqlmodel import Session, col

from app.models.setting import LLMModel, LLMProvider, LLMSetting, ModelRole


def get_role(session: Session, role: ModelRole) -> int:
    """Return the model ID configured for the given role."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        raise ValueError
    return getattr(setting, f'{role.value}_id')


def get_all_with_details(session: Session) -> dict:
    """Return all role settings with their associated model and provider details."""
    setting = session.get(LLMSetting, 1)

    result = {}
    for role in ModelRole:
        model_id = getattr(setting, f'{role.value}_id') if setting else None
        if model_id is not None:
            stmt = (
                select(
                    col(LLMModel.id),
                    col(LLMModel.name).label('model_name'),
                    col(LLMProvider.name).label('provider_name'),
                )
                .join(LLMProvider, col(LLMModel.provider_id) == LLMProvider.id)
                .where(LLMModel.id == model_id)
            )
            # pyrefly: ignore [no-matching-overload]
            row = session.exec(stmt).one_or_none()
            if row:
                result[role.value] = {
                    'model_id': model_id,
                    'model_name': row.model_name,
                    'provider_name': row.provider_name,
                }
            else:
                result[role.value] = {'model_id': None, 'model_name': 'Unknown', 'provider_name': 'Unknown'}
        else:
            result[role.value] = {'model_id': None, 'model_name': 'Not configured', 'provider_name': 'Not configured'}

    return result


def update(session: Session, role: ModelRole, model_id: int) -> LLMSetting:
    """Update the model assignment for the given role and return the updated setting."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        setting = LLMSetting(id=1)
        session.add(setting)
    setattr(setting, f'{role.value}_id', model_id)
    session.commit()
    session.refresh(setting)
    return setting
