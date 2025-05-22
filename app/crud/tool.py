from typing import Optional

from sqlalchemy import Select
from sqlmodel import Session, select, col
from sqlalchemy.orm import selectinload
from app.models.tools import ToolData, SimpleToolPublic


def insert(session: Session, tool: ToolData):
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool


def get_all(session: Session, limit: int, offset: int) -> list[SimpleToolPublic]:
    statement: Select = (
        select(ToolData)
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    tools = session.exec(statement).all()

    result = []
    for tool in tools:
        tool_public = SimpleToolPublic.from_orm(tool)
        result.append(tool_public)
    return result


def get_all_by_group(
    session: Session, parent_id: Optional[int], limit: int, offset: int
) -> list[SimpleToolPublic]:
    statement: Select = (
        select(ToolData)
        .where(ToolData.group_id == parent_id)
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    tools = session.exec(statement).all()

    result = []
    for tool in tools:
        tool_public = SimpleToolPublic.from_orm(tool)
        result.append(tool_public)
    return result


def get_by_name_fuzzy(
    session: Session, name: str, limit: int, offset: int
) -> list[SimpleToolPublic]:
    statement: Select = (
        select(ToolData)
        .where(col(ToolData.name).ilike(f'%{name}%'))
        .options(selectinload(ToolData.tags))  # type: ignore
        .offset(offset)
        .limit(limit)
    )
    tools = session.exec(statement).all()

    result = []
    for tool in tools:
        tool_public = SimpleToolPublic.from_orm(tool)
        result.append(tool_public)
    return result


def get(session: Session, tool_uid: str) -> ToolData:
    statement: Select = select(ToolData).where(ToolData.uid == tool_uid)
    return session.exec(statement).one()
