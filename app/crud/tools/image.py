from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND

from app.models.tools import ToolImage
from app.schemas.tools import ImageUpdate

_FUZZY_SIMILARITY_THRESHOLD = 0.3


def insert(session: Session, image: ToolImage) -> ToolImage:
    """Insert a new tool image into the database."""
    session.add(image)
    session.commit()
    session.refresh(image)

    return image


def insert_all(session: Session, images: list[ToolImage]) -> list[ToolImage]:
    """Insert multiple tool images into the database."""
    session.add_all(images)
    session.commit()

    for image in images:
        session.refresh(image)

    return images


def delete(session: Session, image_uid: str) -> None:
    """Delete a tool image by its UID."""
    image = get(session, image_uid)
    if image:
        session.delete(image)
        session.commit()


def drop_all(session: Session) -> int:
    """Truncate the tool image table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolImage)
    count = session.exec(count_stmt).one()

    if count == 0:
        return 0

    table_name = ToolImage.__tablename__
    # pyrefly: ignore [no-matching-overload]
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def update(session: Session, image_uid: str, image_update: ImageUpdate) -> ToolImage | None:
    """部分更新镜像信息."""
    db_image = get(session, image_uid)
    if not db_image:
        return None

    update_data = image_update.model_dump(exclude_unset=True)
    db_image.sqlmodel_update(update_data)

    session.add(db_image)
    session.commit()
    session.refresh(db_image)
    return db_image


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[ToolImage], int]:
    """Return a page of tool images together with the total count."""
    page_stmt = select(ToolImage).order_by(col(ToolImage.name)).limit(limit).offset(offset)
    count_stmt = select(func.count()).select_from(ToolImage)
    return session.exec(page_stmt).all(), session.exec(count_stmt).one()


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> Sequence[ToolImage]:
    """Return tool images whose names are similar to the given string, ordered by similarity."""
    stmt = (
        select(ToolImage)
        .where(func.similarity(ToolImage.name, name) > _FUZZY_SIMILARITY_THRESHOLD)
        .order_by(func.similarity(ToolImage.name, name).desc())
        .offset(offset)
        .limit(limit)
    )

    return session.exec(stmt).all()


def get_page_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> tuple[Sequence[ToolImage], int]:
    """Return a page of fuzzy-matched tool images together with the total match count."""
    stmt = (
        select(func.count())
        .select_from(ToolImage)
        .where(func.similarity(ToolImage.name, name) > _FUZZY_SIMILARITY_THRESHOLD)
    )
    return get_by_name_fuzzy(session, name, limit, offset), session.exec(stmt).one()


def get(session: Session, image_uid: str) -> ToolImage:
    """Return a tool image by its UID."""
    stmt = select(ToolImage).where(ToolImage.uid == image_uid)
    image = session.exec(stmt).one_or_none()
    if image is None:
        raise HTTPException(HTTP_404_NOT_FOUND, 'Chat session not found')
    return image


def get_by_name(session: Session, name: str) -> ToolImage | None:
    """Return a tool image by its name."""
    stmt = select(ToolImage).where(ToolImage.name == name)
    return session.exec(stmt).one_or_none()
