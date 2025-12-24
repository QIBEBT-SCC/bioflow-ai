from typing import Optional

from sqlmodel import Session, select

from app.models.setting import LLMModel, LLMProvider, LLMSetting


def insert(session: Session, setting: LLMSetting):
    session.add(setting)
    session.commit()
    session.refresh(setting)

    return setting


def get(session: Session, key: str) -> Optional[LLMSetting]:
    statement = select(LLMSetting).where(LLMSetting.key == key)

    return session.exec(statement).one_or_none()


def get_all(session: Session):
    statement = select(LLMSetting)

    return session.exec(statement).all()


def get_all_with_details(session: Session) -> dict:
    """
    获取所有 LLM 设置，包含 model_name 和 provider_name

    Returns:
        包含所有设置的字典，key 为 setting_key，value 包含 model_id, model_name, provider_name
    """
    # 定义所有的设置 keys
    setting_keys = [
        'chat_model',
        'vision_model',
        'agent_model',
        'coding_model',
        'simple_model',
        'long_context_model',
        'high_performance_model',
    ]

    result = {}

    for key in setting_keys:
        # 查询设置并 JOIN model 和 provider
        statement = (
            select(
                LLMSetting.model_id,
                LLMModel.name.label('model_name'),
                LLMProvider.name.label('provider_name'),
            )
            .where(LLMSetting.key == key)
            .join(LLMModel, LLMSetting.model_id == LLMModel.id, isouter=True)
            .join(LLMProvider, LLMModel.provider_id == LLMProvider.id, isouter=True)
        )

        row = session.exec(statement).one_or_none()

        if row:
            result[key] = {
                'model_id': row.model_id,
                'model_name': row.model_name or 'Unknown',
                'provider_name': row.provider_name or 'Unknown',
            }
        else:
            # 如果设置不存在，返回空值
            result[key] = {
                'model_id': None,
                'model_name': 'Not configured',
                'provider_name': 'Not configured',
            }

    return result


def update(session: Session, key: str, model_id: int) -> Optional[LLMSetting]:
    """
    更新 LLM 设置

    Args:
        session: 数据库会话
        key: 设置的 key
        model_id: 新的 model_id

    Returns:
        更新后的设置对象，如果设置不存在则返回 None
    """
    setting = get(session, key)

    if setting is None:
        # 如果设置不存在，创建新的设置
        setting = LLMSetting(key=key, model_id=model_id)
        return insert(session, setting)

    # 更新设置
    setting.model_id = model_id
    session.add(setting)
    session.commit()
    session.refresh(setting)

    return setting
