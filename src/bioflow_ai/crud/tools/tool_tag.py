from collections.abc import Sequence

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from bioflow_ai.models.tools import ToolTag


def insert(session: Session, tag: ToolTag) -> ToolTag:
    """Insert a new tool tag into the database."""
    session.add(tag)
    session.flush()
    session.refresh(tag)

    return tag


def insert_all(session: Session, tags: list[ToolTag]) -> list[ToolTag]:
    """Insert multiple tool tags into the database."""
    session.add_all(tags)
    session.flush()

    for tag in tags:
        session.refresh(tag)

    return tags


def drop_all(session: Session) -> int:
    """Truncate the tool tag table and return the number of deleted rows."""
    count_query = select(func.count()).select_from(ToolTag)
    count = session.scalar(count_query) or 0

    if count == 0:
        return 0

    table_name = ToolTag.__tablename__
    session.execute(text(f'TRUNCATE TABLE {table_name} CASCADE'))

    return count


def get_all(session: Session) -> Sequence[ToolTag]:
    """Return all tool tags ordered by ID."""
    query = select(ToolTag).order_by(ToolTag.id)

    return session.scalars(query).all()


def get_all_by_id(session: Session, ids: list[int]) -> Sequence[ToolTag]:
    """Return all tool tags whose IDs are in the given list."""
    query = select(ToolTag).where(ToolTag.id.in_(ids))
    return session.scalars(query).all()


def get_by_name(session: Session, name: str) -> ToolTag | None:
    """Return a tool tag by its name, or None if not found."""
    query = select(ToolTag).where(ToolTag.name == name)
    return session.scalar(query)
