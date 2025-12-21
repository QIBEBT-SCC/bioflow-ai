from typing import Optional

from sqlmodel import Session, select

from app.models.setting import LLMSetting


def insert(session: Session, setting: LLMSetting):
    session.add(setting)
    session.commit()
    session.refresh(setting)

    return setting


def get(session: Session, key: str) -> Optional[LLMSetting]:
    statement = select(LLMSetting).where(LLMSetting.key == key)

    return session.exec(statement).one_or_none()
