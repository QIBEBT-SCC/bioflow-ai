from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.tools import ToolTag


def insert(session: Session, tag: ToolTag) -> ToolTag:
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def insert_list(session: Session, tags: list[ToolTag]) -> list[ToolTag]:
    session.add_all(tags)
    session.commit()

    for tag in tags:
        session.refresh(tag)

    return tags


def get_all(session: Session) -> list[ToolTag]:
    statement = select(ToolTag)

    return session.exec(statement).all()


def get_all_by_id(session: Session, ids: list[int]) -> list[ToolTag]:
    statement: Select = select(ToolTag).where(col(ToolTag.id).in_(ids))
    return session.exec(statement).all()


def get_by_name(session: Session, name: str):
    statement: Select = select(ToolTag).where(ToolTag.name == name)
    return session.exec(statement).one_or_none()


def ai_gen_tag(session: Session) -> ToolTag:
    tag = get_by_name(session, 'AI Gen')
    if tag:
        return tag

    new_tag = insert(session, ToolTag(name='AI Gen'))
    return new_tag
