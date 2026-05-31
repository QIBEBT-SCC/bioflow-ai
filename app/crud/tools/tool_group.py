from collections.abc import Sequence

from sqlalchemy import func, text
from sqlalchemy import select as as_select
from sqlmodel import Session, col, select

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
    count_stmt = select(func.count()).select_from(ToolGroup)
    count = session.exec(count_stmt).one()

    if count == 0:
        return 0

    table_name = ToolGroup.__tablename__
    # pyrefly: ignore [no-matching-overload]
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def get(session: Session, group_id: int) -> ToolGroup | None:
    """Return a tool group by its ID, or None if not found."""
    stmt = select(ToolGroup).where(ToolGroup.id == group_id)
    return session.exec(stmt).one_or_none()


def get_by_name(session: Session, name: str) -> ToolGroup | None:
    """Return a tool group by its name, or None if not found."""
    stmt = select(ToolGroup).where(ToolGroup.name == name)
    return session.exec(stmt).one_or_none()


def get_all(session: Session) -> Sequence[ToolGroup]:
    """Return all tool groups."""
    stmt = select(ToolGroup)
    return session.exec(stmt).all()


def get_all_with_tool_counts(session: Session) -> list[dict]:
    """获取所有工具分组及其对应的工具数量."""
    stmt = (
        as_select(
            col(ToolGroup.id),
            col(ToolGroup.name),
            col(ToolGroup.parent_id),
            col(ToolGroup.tag_id),
            func.count(col(ToolDefine.uid)).label('tool_count'),
        )
        .outerjoin(ToolDefine, col(ToolGroup.id) == ToolDefine.group_id)
        .group_by(col(ToolGroup.id), ToolGroup.name, col(ToolGroup.parent_id))
        .order_by(col(ToolGroup.id))
    )

    # pyrefly: ignore [no-matching-overload]
    results = session.exec(stmt).all()

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
