from typing import Optional

from sqlalchemy import Select, func
from sqlmodel import Session, select, col

from app.models.tools import ToolImage


def insert(session: Session, image: ToolImage) -> ToolImage:
    session.add(image)
    session.commit()
    session.refresh(image)

    return image


def insert_all(session: Session, images: list[ToolImage]):
    session.add_all(images)
    session.commit()

    for image in images:
        session.refresh(image)

    return images


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int):
    query: Select = (
        select(ToolImage)
        .where(func.similarity(ToolImage.name, name) > 0.1)
        .order_by(func.similarity(ToolImage.name, name).desc())
        .offset(offset)
        .limit(limit)
    )

    return session.exec(query).all()


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(ToolImage).order_by(col(ToolImage.id)).limit(limit).offset(offset)
    return session.exec(query).all()


def get(session: Session, image_id: int) -> Optional[ToolImage]:
    query: Select = select(ToolImage).where(ToolImage.id == image_id)
    return session.exec(query).one_or_none()
