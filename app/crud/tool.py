from sqlmodel import Session

from app.models.tools import ToolData


def insert(session: Session, tool: ToolData):
    session.add(tool)
    session.commit()
    session.refresh(tool)

    return tool
