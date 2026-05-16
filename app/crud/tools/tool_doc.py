from typing import Optional

from sqlalchemy import Select, func, text
from sqlmodel import Session, col, select

from app.models.tools import ToolHelpDoc
from app.schemas.tools import DocUpdate


def insert(session: Session, doc: ToolHelpDoc) -> ToolHelpDoc:
    """Insert a new tool help document into the database."""
    session.add(doc)
    session.commit()
    session.refresh(doc)

    return doc


def insert_all(session: Session, docs: list[ToolHelpDoc]) -> list[ToolHelpDoc]:
    """Insert multiple tool help documents into the database."""
    session.add_all(docs)
    session.commit()

    for doc in docs:
        session.refresh(doc)

    return docs


def delete(session: Session, doc_uid: str) -> None:
    """Delete a tool help document by its UID."""
    doc = get(session, doc_uid)
    if doc:
        session.delete(doc)
        session.commit()


def drop_all(session: Session) -> int:
    """Truncate the tool help doc table and return the number of deleted rows."""
    count_query = select(func.count()).select_from(ToolHelpDoc)
    count = session.exec(count_query).one()

    if count == 0:
        return 0

    table_name = ToolHelpDoc.__tablename__
    session.exec(text(f'TRUNCATE TABLE {table_name} CASCADE'))
    session.commit()

    return count


def update(session: Session, image_uid: str, doc_update: DocUpdate) -> Optional[ToolHelpDoc]:
    """部分更新镜像信息."""
    doc = get(session, image_uid)
    if not doc:
        return None

    update_data = doc_update.model_dump(exclude_unset=True)
    doc.sqlmodel_update(update_data)

    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def get_all(session: Session, limit: int, offset: int) -> list[ToolHelpDoc]:
    """Return a paginated list of all tool help documents ordered by UID."""
    query: Select = select(ToolHelpDoc).order_by(col(ToolHelpDoc.uid)).limit(limit).offset(offset)
    return session.exec(query).all()


def get(session: Session, doc_uid: str) -> ToolHelpDoc:
    """Return a tool help document by its UID."""
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.uid == doc_uid)

    return session.exec(query).one()


def get_by_image(session: Session, image_uid: str) -> list[ToolHelpDoc]:
    """Return all tool help documents associated with the given image UID."""
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.image_uid == image_uid)

    return session.exec(query).all()


def get_by_command(session: Session, help_command: str) -> Optional[ToolHelpDoc]:
    """Return a tool help document by its help command, or None if not found."""
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.help_command == help_command)

    return session.exec(query).one_or_none()


def get_counts(session: Session) -> int:
    """Return the total number of tool help documents."""
    query: Select = select(func.count()).select_from(ToolHelpDoc)

    return session.exec(query).one()
