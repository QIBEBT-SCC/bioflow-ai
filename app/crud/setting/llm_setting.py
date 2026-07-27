from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.setting import LLMModel, LLMProvider, LLMSetting, ModelRole


def get_role(session: Session, role: ModelRole) -> int | None:
    """Return the model ID configured for the given role, when present."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        return None
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
                    LLMModel.id,
                    LLMModel.display_name,
                    LLMModel.name,
                    LLMProvider.name.label('provider_name'),
                )
                .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
                .where(LLMModel.id == model_id)
            )
            row = session.execute(stmt).one_or_none()
            if row:
                result[role.value] = {
                    'model_id': model_id,
                    'display_name': row.display_name,
                    'name': row.name,
                    'provider_name': row.provider_name,
                }
            else:
                result[role.value] = {
                    'model_id': None,
                    'display_name': 'Unknown',
                    'name': 'Unknown',
                    'provider_name': 'Unknown',
                }
        else:
            result[role.value] = {
                'model_id': None,
                'display_name': 'Not configured',
                'name': 'Not configured',
                'provider_name': 'Not configured',
            }

    return result


def update(session: Session, role: ModelRole, model_id: int) -> LLMSetting:
    """Stage a model assignment for one role."""
    setting = session.get(LLMSetting, 1)
    if setting is None:
        setting = LLMSetting(id=1)
        session.add(setting)
    setattr(setting, f'{role.value}_id', model_id)
    session.flush()
    session.refresh(setting)
    return setting
