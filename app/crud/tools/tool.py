from typing import Optional

from sqlalchemy import Select, func
from sqlmodel import Session, select, col
from sqlalchemy.orm import selectinload
from app.models.tools import ToolData


def insert(session: Session, tool: ToolData):
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def insert_all(session: Session, tools: list[ToolData]):
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


def get_all(session: Session, limit: int, offset: int):
    query: Select = (
        select(ToolData)
        .order_by(col(ToolData.name))
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all_by_group(session: Session, parent_id: Optional[int], limit: int, offset: int):
    query: Select = (
        select(ToolData)
        .where(ToolData.group_id == parent_id)
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int):
    query: Select = (
        select(ToolData)
        .where(func.similarity(ToolData.name, name) > 0.1)
        .order_by(func.similarity(ToolData.name, name).desc())
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get(session: Session, tool_uid: str):
    query: Select = select(ToolData).where(ToolData.uid == tool_uid)
    return session.exec(query).one()
