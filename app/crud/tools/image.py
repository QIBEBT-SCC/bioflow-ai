from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy import func, text, type_coerce
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Session, col, select
from starlette.status import HTTP_404_NOT_FOUND, HTTP_409_CONFLICT

from app.models.tools import ImageConfig, ToolImage
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


def get_by_image_url(session: Session, image_url: str) -> ToolImage | None:
    """Return a tool image by Docker image URL.

    If the URL includes a tag, the tag is matched too. If the URL does not include a tag,
    only registry/namespace/repository are matched, and multiple matches are treated as an
    ambiguous image selection.
    """
    image_config = ImageConfig.from_str(image_url)
    has_tag = _has_explicit_tag(image_url)
    image_field = type_coerce(col(ToolImage.image), JSONB)

    conditions = [
        image_field['registry'].astext == image_config.registry,
        image_field['namespace'].astext == image_config.namespace,
        image_field['repository'].astext == image_config.repository,
    ]
    if has_tag:
        conditions.append(image_field['tag'].astext == image_config.tag)

    stmt = select(ToolImage).where(*conditions).limit(2)
    images = list(session.exec(stmt).all())

    if not images:
        return None
    if len(images) > 1:
        image_ref = (
            image_config.get_image()
            if has_tag
            else f'{image_config.registry}/{image_config.namespace}/{image_config.repository}'
        )
        raise HTTPException(
            HTTP_409_CONFLICT,
            f'Multiple image records match "{image_ref}". Please provide a full image URL with tag.',
        )

    return images[0]


def get_by_image_ref(session: Session, image_ref: str) -> ToolImage | None:
    """Return a tool image by full or partial Docker image reference.

    Supported references include repository, namespace/repository, full
    registry/namespace/repository, and any of those forms with an explicit tag.
    Ambiguous matches raise a conflict error.
    """
    image_name, tag = _split_image_tag(image_ref)
    url_group = image_name.split('/')
    image_field = type_coerce(col(ToolImage.image), JSONB)

    match len(url_group):
        case 3:
            conditions = [
                image_field['registry'].astext == url_group[0],
                image_field['namespace'].astext == url_group[1],
                image_field['repository'].astext == url_group[2],
            ]
        case 2:
            conditions = [
                image_field['namespace'].astext == url_group[0],
                image_field['repository'].astext == url_group[1],
            ]
        case 1:
            conditions = [image_field['repository'].astext == url_group[0]]
        case _:
            err_msg = 'Incorrect image format. It MUST be like `[registry/][namespace/]<repository>[:tag]`'
            raise ValueError(err_msg)

    if tag is not None:
        conditions.append(image_field['tag'].astext == tag)

    stmt = select(ToolImage).where(*conditions).limit(2)
    images = list(session.exec(stmt).all())

    if not images:
        return None
    if len(images) > 1:
        raise HTTPException(
            HTTP_409_CONFLICT,
            f'Multiple image records match "{image_ref}". Please provide a more specific image reference.',
        )

    return images[0]


def _has_explicit_tag(image_url: str) -> bool:
    """Return whether a Docker image string explicitly includes a tag."""
    return ':' in image_url.rsplit('/', 1)[-1]


def _split_image_tag(image_ref: str) -> tuple[str, str | None]:
    """Split a Docker image reference into name and optional explicit tag."""
    if not _has_explicit_tag(image_ref):
        return image_ref, None
    image_name, tag = image_ref.rsplit(':', 1)
    return image_name, tag


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


def get_unique_by_name(session: Session, name: str) -> ToolImage | None:
    """Return a unique tool image by tool name, or raise on ambiguity."""
    stmt = select(ToolImage).where(ToolImage.name == name).limit(2)
    images = list(session.exec(stmt).all())

    if not images:
        return None
    if len(images) > 1:
        raise HTTPException(
            HTTP_409_CONFLICT,
            f'Multiple image records match tool name "{name}". Please provide an image repository reference.',
        )

    return images[0]
