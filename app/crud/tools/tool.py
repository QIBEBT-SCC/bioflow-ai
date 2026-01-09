from typing import Optional

from loguru import logger
from sqlalchemy import Select, func
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from app.models.tools import ToolDefine, ToolTag


def insert(session: Session, tool: ToolDefine):
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def insert_all(session: Session, tools: list[ToolDefine]):
    session.add_all(tools)
    session.commit()

    for tool in tools:
        session.refresh(tool)

    return tools


def delete(session: Session, tool_uid: str):
    tool = get(session, tool_uid)
    if tool:
        session.delete(tool)
        session.commit()


def update(session: Session, tool: ToolDefine):
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def update_partial(session: Session, tool_uid: str, tool_update):
    """部分更新工具配置"""
    from app.crud.tools import tool_tag as tag_crud

    db_tool = get(session, tool_uid)
    if not db_tool:
        return None

    # 准备更新数据
    update_data = tool_update.model_dump(exclude_unset=True)
    db_tool.sqlmodel_update(update_data)

    # 处理标签更新
    if tool_update.tags is not None:
        exist_ids: list[int] = [tag.id for tag in tool_update.tags if tag.id != -1]
        tags = tag_crud.get_all_by_id(session, exist_ids)
        db_tool.tags = tags

    session.add(db_tool)
    session.commit()
    session.refresh(db_tool)

    return db_tool


def get_counts(session: Session):
    query: Select = select(func.count()).select_from(ToolDefine)

    return session.exec(query).one()


def get_all(session: Session, limit: int, offset: int):
    query: Select = (
        select(ToolDefine)
        .order_by(col(ToolDefine.name))
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all_by_group(session: Session, group_id: Optional[int], limit: int, offset: int):
    query: Select = (
        select(ToolDefine)
        .where(ToolDefine.group_id == group_id)
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all_by_image(session: Session, image_uid: str):
    query: Select = select(ToolDefine).where(ToolDefine.image_uid == image_uid)

    return session.exec(query).all()


def get_all_by_uid(session: Session, uids: list[str]):
    query: Select = select(ToolDefine).where(col(ToolDefine.uid).in_(uids))
    return session.exec(query).all()


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int):
    query: Select = (
        select(ToolDefine)
        .where(func.similarity(ToolDefine.name, name) > 0.1)
        .order_by(func.similarity(ToolDefine.name, name).desc())
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get(session: Session, tool_uid: str) -> Optional[ToolDefine]:
    query: Select = select(ToolDefine).where(ToolDefine.uid == tool_uid)
    return session.exec(query).one_or_none()
