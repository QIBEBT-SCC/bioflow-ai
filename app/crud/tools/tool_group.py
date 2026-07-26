from collections.abc import Sequence

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.tools import ToolDefine, ToolGroup


def insert(session: Session, group: ToolGroup) -> ToolGroup:
    """Insert a new tool group into the database."""
    session.add(group)
    session.flush()
    session.refresh(group)

    return group


def insert_all(session: Session, groups: list[ToolGroup]) -> list[ToolGroup]:
    """Insert multiple tool groups into the database."""
    session.add_all(groups)
    session.flush()

    for group in groups:
        session.refresh(group)

    return groups


def drop_all(session: Session) -> int:
    """Truncate the tool group table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolGroup)
    count = session.scalar(count_stmt) or 0

    if count == 0:
        return 0

    table_name = ToolGroup.__tablename__
    session.execute(text(f'TRUNCATE TABLE {table_name} CASCADE'))

    return count


def get(session: Session, group_id: int) -> ToolGroup | None:
    """Return a tool group by its ID, or None if not found."""
    stmt = select(ToolGroup).where(ToolGroup.id == group_id)
    return session.scalar(stmt)


def get_by_name(session: Session, name: str) -> ToolGroup | None:
    """Return a tool group by its name, or None if not found."""
    stmt = select(ToolGroup).where(ToolGroup.name == name)
    return session.scalar(stmt)


def get_all(session: Session) -> Sequence[ToolGroup]:
    """Return all tool groups."""
    stmt = select(ToolGroup)
    return session.scalars(stmt).all()


def get_all_with_tool_counts(session: Session) -> list[dict]:
    """获取所有工具分组及其对应的工具数量."""
    stmt = (
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

    return [dict(result) for result in session.execute(stmt).mappings()]
