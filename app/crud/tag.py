from typing import Sequence

from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.project import Tag


def insert(session: Session, tag: Tag) -> Tag:
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def get_all(session: Session) -> Sequence[Tag]:
    statement = select(Tag)
    return session.exec(statement).all()


def get_all_by_id(session: Session, ids: list[int]) -> Sequence[Tag]:
    statement: Select = select(Tag).where(col(Tag.id).in_(ids))
    return session.exec(statement).all()
