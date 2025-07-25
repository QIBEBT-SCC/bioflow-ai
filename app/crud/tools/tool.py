from typing import Optional

from sqlalchemy import Select, func
from sqlmodel import Session, select, col
from sqlalchemy.orm import selectinload
from app.models.tools import ToolDefine


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


def delete(session: Session, tool_id: int):
    tool = get(session, tool_id)
    if tool:
        session.delete(tool)
        session.commit()


def get_all(session: Session, limit: int, offset: int):
    query: Select = (
        select(ToolDefine)
        .order_by(col(ToolDefine.id))
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all_by_group(session: Session, parent_id: Optional[int], limit: int, offset: int):
    query: Select = (
        select(ToolDefine)
        .where(ToolDefine.group_id == parent_id)
        .options(selectinload(ToolDefine.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all_by_uid(session: Session, ids: list[int]):
    query: Select = select(ToolDefine).where(col(ToolDefine.id).in_(ids))
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


def get(session: Session, tool_id: int):
    query: Select = select(ToolDefine).where(ToolDefine.id == tool_id)
    return session.exec(query).one()
