from collections.abc import Sequence

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload

from app.models.tools import ToolHelpDoc
from app.schemas.tools import DocUpdate
from app.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, doc: ToolHelpDoc) -> ToolHelpDoc:
    """Insert a new tool help document into the database."""
    session.add(doc)
    session.flush()
    session.refresh(doc)

    return doc


def insert_all(session: Session, docs: list[ToolHelpDoc]) -> list[ToolHelpDoc]:
    """Insert multiple tool help documents into the database."""
    session.add_all(docs)
    session.flush()

    for doc in docs:
        session.refresh(doc)

    return docs


def delete(session: Session, doc_uid: UUIDLike) -> bool:
    """Delete a tool help document by its UID and report whether it existed."""
    doc = get(session, doc_uid)
    if doc is None:
        return False
    session.delete(doc)
    session.flush()
    return True


def drop_all(session: Session) -> int:
    """Truncate the tool help doc table and return the number of deleted rows."""
    count_stmt = select(func.count()).select_from(ToolHelpDoc)
    count = session.scalar(count_stmt) or 0

    if count == 0:
        return 0

    table_name = ToolHelpDoc.__tablename__
    session.execute(text(f'TRUNCATE TABLE {table_name} CASCADE'))

    return count


def update(session: Session, doc_uid: UUIDLike, doc_update: DocUpdate) -> ToolHelpDoc | None:
    """Update a help document, or return None if it does not exist."""
    doc = get(session, doc_uid)
    if doc is None:
        return None

    update_data = doc_update.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        setattr(doc, field_name, value)

    session.add(doc)
    session.flush()
    session.refresh(doc)
    return doc


def get_all(session: Session, limit: int, offset: int) -> Sequence[ToolHelpDoc]:
    """Return a paginated list of all tool help documents ordered by UID."""
    stmt = select(ToolHelpDoc).order_by(ToolHelpDoc.uid).limit(limit).offset(offset)
    return session.scalars(stmt).all()


def get(session: Session, doc_uid: UUIDLike) -> ToolHelpDoc | None:
    """Return a tool help document by its UID, or None if it does not exist."""
    stmt = select(ToolHelpDoc).where(ToolHelpDoc.uid == as_uuid(doc_uid)).options(selectinload(ToolHelpDoc.image))

    return session.scalar(stmt)


def get_by_image(session: Session, image_uid: UUIDLike) -> Sequence[ToolHelpDoc]:
    """Return all tool help documents associated with the given image UID."""
    stmt = select(ToolHelpDoc).where(ToolHelpDoc.image_uid == as_uuid(image_uid))

    return session.scalars(stmt).all()


def get_by_command(
    session: Session,
    help_command: str,
    image_uid: UUIDLike,
) -> ToolHelpDoc | None:
    """Return a tool help document by its help command, or None if not found."""
    stmt = select(ToolHelpDoc).where(
        ToolHelpDoc.help_command == help_command,
        ToolHelpDoc.image_uid == as_uuid(image_uid),
    )

    return session.scalar(stmt)


def get_counts(session: Session) -> int:
    """Return the total number of tool help documents."""
    stmt = select(func.count()).select_from(ToolHelpDoc)

    return session.scalar(stmt) or 0
