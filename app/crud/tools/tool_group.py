from typing import Optional

from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.tools import ToolGroup


def insert(session: Session, group: ToolGroup) -> ToolGroup:
    session.add(group)
    session.commit()
    session.refresh(group)

    return group


def insert_all(session: Session, groups: list[ToolGroup]):
    session.add_all(groups)
    session.commit()

    for group in groups:
        session.refresh(group)

    return groups


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(ToolGroup).limit(limit).offset(offset)
    return session.exec(query).all()


def get_all_recursive(session: Session, parent_id: Optional[int] = None):
    if parent_id is None:
        base: Select = select(ToolGroup)
        return session.exec(base).all()

    # 1. 基础 CTE，查找指定父节点的直接子节点
    base = select(ToolGroup).where(ToolGroup.parent_id == parent_id)
    group_cte = base.cte(name='group_tree', recursive=True)

    # 2. 递归部分，查找所有子孙节点
    recursive = select(ToolGroup).where(ToolGroup.parent_id == group_cte.c.id)
    group_cte = group_cte.union_all(recursive)

    # 3. 查询所有结果
    stmt = select(group_cte.c.id)
    group_ids = session.exec(stmt).all()

    # 4. 基于id查询实际对象
    query: Select = select(ToolGroup).where(col(ToolGroup.id).in_(group_ids))
    return session.exec(query).all()


def get_by_name(session: Session, name: str) -> Optional[ToolGroup]:
    query: Select = select(ToolGroup).where(ToolGroup.name == name)
    return session.exec(query).one_or_none()
