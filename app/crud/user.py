from typing import Optional

from loguru import logger
from sqlalchemy import Select
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models.user import User


class UserExistError(ValueError):
    pass


def insert(session: Session, user: User) -> None:
    try:
        logger.debug(f'新增用户请求 ({User.model_validate(user)})')
        session.add(user)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise UserExistError from e
        else:
            raise e


def get(session: Session, email: str) -> Optional[User]:
    statement: Select = select(User).where(User.email == email)
    return session.exec(statement).first()
