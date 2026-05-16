from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models.user import User

if TYPE_CHECKING:
    from sqlalchemy import Select


class UserExistError(ValueError):
    pass


def insert(session: Session, user: User) -> None:
    """Insert a new user into the database."""
    try:
        logger.debug(f'新增用户请求 ({User.model_validate(user)})')
        session.add(user)
        session.commit()
    except IntegrityError as e:
        if 'UNIQUE' in e.args[0]:
            raise UserExistError from e
        raise


def get(session: Session, email: str) -> User:
    """Return a user by email address."""
    statement: Select = select(User).where(User.email == email)
    return session.exec(statement).one()
