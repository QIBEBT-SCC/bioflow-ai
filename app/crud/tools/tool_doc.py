from typing import Optional

from sqlalchemy import Select, func
from sqlmodel import Session, col, select

from app.models.tools import ToolHelpDoc
from app.schemas.tools import DocUpdate


def insert(session: Session, doc: ToolHelpDoc) -> ToolHelpDoc:
    session.add(doc)
    session.commit()
    session.refresh(doc)

    return doc


def insert_all(session: Session, docs: list[ToolHelpDoc]):
    session.add_all(docs)
    session.commit()

    for doc in docs:
        session.refresh(doc)

    return docs


def delete(session: Session, doc_uid: str):
    doc = get(session, doc_uid)
    if doc:
        session.delete(doc)
        session.commit()


def update(session: Session, image_uid: str, doc_update: DocUpdate):
    """部分更新镜像信息"""
    doc = get(session, image_uid)
    if not doc:
        return None

    update_data = doc_update.model_dump(exclude_unset=True)
    doc.sqlmodel_update(update_data)

    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(ToolHelpDoc).order_by(col(ToolHelpDoc.uid)).limit(limit).offset(offset)
    return session.exec(query).all()


def get(session: Session, doc_uid: str) -> ToolHelpDoc:
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.uid == doc_uid)

    return session.exec(query).one()


def get_by_image(session: Session, image_uid: str):
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.image_uid == image_uid)

    return session.exec(query).all()


def get_by_command(session: Session, help_command: str) -> Optional[ToolHelpDoc]:
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.help_command == help_command)

    return session.exec(query).one_or_none()


def get_counts(session: Session):
    query: Select = select(func.count()).select_from(ToolHelpDoc)

    return session.exec(query).one()
