from sqlalchemy import Select, func, text
from sqlmodel import Session, col, select

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


def drop_all(session: Session) -> int:
    count_query = select(func.count()).select_from(ToolTag)
    count = session.exec(count_query).one()

    if count == 0:
        return 0

    table_name = ToolTag.__tablename__
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def get_all(session: Session):
    query = select(ToolTag).order_by(col(ToolTag.id))

    return session.exec(query).all()


def get_all_by_id(session: Session, ids: list[int]):
    query: Select = select(ToolTag).where(col(ToolTag.id).in_(ids))
    return session.exec(query).all()


def get_by_name(session: Session, name: str):
    query: Select = select(ToolTag).where(ToolTag.name == name)
    return session.exec(query).one_or_none()
