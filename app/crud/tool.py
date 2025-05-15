from sqlalchemy import Select
from sqlmodel import Session, select
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
