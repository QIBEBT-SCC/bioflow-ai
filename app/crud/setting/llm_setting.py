from typing import Optional

from sqlmodel import Session, select

from app.models.setting import LLMModel, LLMProvider, LLMSetting, ModelRole


def get_role(session: Session, role: ModelRole) -> int:
    setting = session.get(LLMSetting, 1)
    if setting is None:
        raise ValueError
    return getattr(setting, f'{role.value}_id')


def get_all_with_details(session: Session) -> dict:
    setting = session.get(LLMSetting, 1)

    result = {}
    for role in ModelRole:
        model_id = getattr(setting, f'{role.value}_id') if setting else None
        if model_id is not None:
            stmt = (
                select(LLMModel.id, LLMModel.name.label('model_name'), LLMProvider.name.label('provider_name'))
                .join(LLMProvider, LLMModel.provider_id == LLMProvider.id)
                .where(LLMModel.id == model_id)
            )
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
    setting = session.get(LLMSetting, 1)
    if setting is None:
        setting = LLMSetting(id=1)
        session.add(setting)
    setattr(setting, f'{role.value}_id', model_id)
    session.commit()
    session.refresh(setting)
    return setting
