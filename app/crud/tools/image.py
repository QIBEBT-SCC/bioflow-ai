from typing import Optional

from sqlalchemy import Select
from sqlmodel import Session, select

from app.models.tools import ToolImage


def insert(session: Session, image: ToolImage) -> ToolImage:
    session.add(image)
    session.commit()
    session.refresh(image)

    return image


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(ToolImage).limit(limit).offset(offset)
    return session.exec(query).all()


def get(session: Session, image_id: int) -> Optional[ToolImage]:
    query: Select = select(ToolImage).where(ToolImage.id == image_id)
    return session.exec(query).one_or_none()
