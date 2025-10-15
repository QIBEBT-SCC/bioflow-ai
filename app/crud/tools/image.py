from typing import Optional

from sqlalchemy import Select, func
from sqlmodel import Session, select, col

from app.models.tools import ToolImage
from app.schemas.tools import ImageUpdate


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


def delete(session: Session, image_uid: str):
    image = get(session, image_uid)
    if image:
        session.delete(image)
        session.commit()


def update(session: Session, image_update: ImageUpdate):
    db_image = get(session, image_update.uid)
    if not db_image:
        raise ValueError(f'Image with uid {image_update.uid} not found')

    update_data = image_update.model_dump(exclude_unset=True)
    db_image.sqlmodel_update(update_data)

    session.add(db_image)
    session.commit()
    session.refresh(db_image)
    return db_image


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
    query: Select = select(ToolImage).order_by(col(ToolImage.name)).limit(limit).offset(offset)
    return session.exec(query).all()


def get(session: Session, image_uid: str) -> Optional[ToolImage]:
    query: Select = select(ToolImage).where(ToolImage.uid == image_uid)
    return session.exec(query).one_or_none()


def get_counts(session: Session) -> int:
    query: Select = select(func.count()).select_from(ToolImage)
    return session.exec(query).one()
