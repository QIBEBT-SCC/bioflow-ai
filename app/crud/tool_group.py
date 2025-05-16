from typing import Optional

from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.tools import ToolGroup


def insert(session: Session, group: ToolGroup) -> ToolGroup:
    session.add(group)
    session.commit()
    session.refresh(group)

    return group


def get_all_recursive(session: Session, parent_id: Optional[int] = None) -> list[ToolGroup]:
    if parent_id is None:
        base = select(ToolGroup)
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
    statement: Select = select(ToolGroup).where(col(ToolGroup.id).in_(group_ids))
    return session.exec(statement).all()
