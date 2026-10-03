from collections.abc import Sequence

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.elements import ColumnElement

from bioflow_ai.models.tools import ToolDefine, ToolImage
from bioflow_ai.schemas.tools import ImageUpdate
from bioflow_ai.utils.identifiers import UUIDLike, as_uuid

_FUZZY_SIMILARITY_THRESHOLD = 0.3


def insert(session: Session, image: ToolImage) -> ToolImage:
    """Insert a new tool image into the database."""
    session.add(image)
    session.flush()
    session.refresh(image)

    return image


def insert_all(session: Session, images: list[ToolImage]) -> list[ToolImage]:
    """Insert multiple tool images into the database."""
    session.add_all(images)
    session.flush()

    for image in images:
        session.refresh(image)

    return images


def delete(session: Session, image_uid: UUIDLike) -> bool:
    """Delete a tool image by its UID and report whether it existed."""
    image = get(session, image_uid)
    if image is None:
        return False
    session.delete(image)
    session.flush()
    return True


def drop_all(session: Session) -> int:
    """Truncate the tool image table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolImage)
    count = session.scalar(count_stmt) or 0

    if count == 0:
        return 0

    table_name = ToolImage.__tablename__
    session.execute(text(f'TRUNCATE TABLE {table_name} CASCADE'))

    return count


def update(session: Session, image_uid: UUIDLike, image_update: ImageUpdate) -> ToolImage | None:
    """部分更新镜像信息."""
    db_image = get(session, image_uid)
    if not db_image:
        return None

    update_data = image_update.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(db_image, field_name, value)

    session.add(db_image)
    session.flush()
    session.refresh(db_image)
    return db_image


def get_page(session: Session, limit: int, offset: int) -> tuple[Sequence[ToolImage], int]:
    """Return a page of tool images together with the total count."""
    page_stmt = select(ToolImage).order_by(ToolImage.name).limit(limit).offset(offset)
    count_stmt = select(func.count()).select_from(ToolImage)
    return session.scalars(page_stmt).all(), session.scalar(count_stmt) or 0


def get_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> Sequence[ToolImage]:
    """Return tool images whose names or aliases fuzzy-match the query."""
    similarity = _name_or_alias_similarity(name)
    stmt = (
        select(ToolImage)
        .where(similarity > _FUZZY_SIMILARITY_THRESHOLD)
        .order_by(similarity.desc(), ToolImage.name)
        .offset(offset)
        .limit(limit)
    )

    return session.scalars(stmt).all()


def get_by_ref(session: Session, image_ref: str) -> ToolImage | None:
    """Return the image with an exact generated reference."""
    stmt = select(ToolImage).where(ToolImage.ref == image_ref)
    return session.scalar(stmt)


def get_page_by_name_fuzzy(session: Session, name: str, limit: int, offset: int) -> tuple[Sequence[ToolImage], int]:
    """Return a page of fuzzy-matched tool images together with the total match count."""
    similarity = _name_or_alias_similarity(name)
    stmt = select(func.count()).select_from(ToolImage).where(similarity > _FUZZY_SIMILARITY_THRESHOLD)
    return get_by_name_fuzzy(session, name, limit, offset), session.scalar(stmt) or 0


def _name_or_alias_similarity(name: str) -> ColumnElement[float]:
    """Return the best equally weighted similarity across an image name and each alias."""
    alias_rows = func.unnest(ToolImage.aliases).alias('image_alias')
    alias_similarity = select(func.max(func.similarity(alias_rows.column, name))).correlate(ToolImage).scalar_subquery()
    return func.greatest(
        func.similarity(ToolImage.name, name),
        func.coalesce(alias_similarity, 0.0),
    )


def get(session: Session, image_uid: UUIDLike) -> ToolImage | None:
    """Return a tool image by its UID, or None if it does not exist."""
    stmt = (
        select(ToolImage)
        .where(ToolImage.uid == as_uuid(image_uid))
        .options(
            selectinload(ToolImage.tools).selectinload(ToolDefine.tags),
            selectinload(ToolImage.tools).selectinload(ToolDefine.image),
        )
    )
    return session.scalar(stmt)


def get_by_name(session: Session, name: str) -> ToolImage | None:
    """Return a tool image by its name."""
    stmt = select(ToolImage).where(ToolImage.name == name)
    return session.scalar(stmt)
