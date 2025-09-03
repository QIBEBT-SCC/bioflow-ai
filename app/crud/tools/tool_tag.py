from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.tools import ToolTag


def insert(session: Session, tag: ToolTag) -> ToolTag:
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def insert_all(session: Session, tags: list[ToolTag]):
    session.add_all(tags)
    session.commit()

    for tag in tags:
        session.refresh(tag)

    return tags


def get_all(session: Session):
    query = select(ToolTag).order_by(col(ToolTag.id))

    return session.exec(query).all()


def get_all_by_id(session: Session, ids: list[int]):
    query: Select = select(ToolTag).where(col(ToolTag.id).in_(ids))
    return session.exec(query).all()


def get_by_name(session: Session, name: str):
    query: Select = select(ToolTag).where(ToolTag.name == name)
    return session.exec(query).one_or_none()
