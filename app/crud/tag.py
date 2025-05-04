from typing import Sequence

from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.project import Tag, TagWithCount


def insert(session: Session, tag: Tag) -> Tag:
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def get_all(session: Session) -> Sequence[TagWithCount]:
    statement = select(Tag, Tag.project_count)
    tags = session.exec(statement).all()
    return [TagWithCount(id=t.id, name=t.name, color=t.color, project_count=c) for t, c in tags]


def get_all_by_id(session: Session, ids: list[int]) -> Sequence[Tag]:
    statement: Select = select(Tag).where(col(Tag.id).in_(ids))
    return session.exec(statement).all()
