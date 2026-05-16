from typing import Optional

from sqlalchemy import Select, func, text
from sqlmodel import Session, select

from app.models.tools import ToolDefine, ToolGroup


def insert(session: Session, group: ToolGroup) -> ToolGroup:
    """Insert a new tool group into the database."""
    session.add(group)
    session.commit()
    session.refresh(group)

    return group


def insert_all(session: Session, groups: list[ToolGroup]) -> list[ToolGroup]:
    """Insert multiple tool groups into the database."""
    session.add_all(groups)
    session.commit()

    for group in groups:
        session.refresh(group)

    return groups


def drop_all(session: Session) -> int:
    """Truncate the tool group table and return the number of deleted rows."""
    count_query = select(func.count()).select_from(ToolGroup)
    count = session.exec(count_query).one()

    if count == 0:
        return 0

    table_name = ToolGroup.__tablename__
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def get(session: Session, group_id: int) -> Optional[ToolGroup]:
    """Return a tool group by its ID, or None if not found."""
    query: Select = select(ToolGroup).where(ToolGroup.id == group_id)
    return session.exec(query).one_or_none()


def get_by_name(session: Session, name: str) -> Optional[ToolGroup]:
    """Return a tool group by its name, or None if not found."""
    query: Select = select(ToolGroup).where(ToolGroup.name == name)
    return session.exec(query).one_or_none()


def get_all(session: Session) -> list[ToolGroup]:
    """Return all tool groups."""
    query: Select = select(ToolGroup)
    return session.exec(query).all()


def get_all_with_tool_counts(session: Session) -> list[dict]:
    """获取所有工具分组及其对应的工具数量."""
    query = (
        select(
            ToolGroup.id,
            ToolGroup.name,
            ToolGroup.parent_id,
            ToolGroup.tag_id,
            func.count(ToolDefine.uid).label('tool_count'),
        )
        .outerjoin(ToolDefine, ToolGroup.id == ToolDefine.group_id)
        .group_by(ToolGroup.id, ToolGroup.name, ToolGroup.parent_id, ToolGroup.tag_id)
        .order_by(ToolGroup.id)
    )

    results = session.exec(query).all()

    return [
        {
            'id': result.id,
            'name': result.name,
            'parent_id': result.parent_id,
            'tag_id': result.tag_id,
            'tool_count': result.tool_count,
        }
        for result in results
    ]
