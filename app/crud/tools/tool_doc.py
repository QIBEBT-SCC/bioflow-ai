from sqlalchemy import Select
from sqlmodel import Session, select, col

from app.models.tools import ToolHelpDoc


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


def get_all(session: Session, limit: int, offset: int):
    query: Select = select(ToolHelpDoc).order_by(col(ToolHelpDoc.id)).limit(limit).offset(offset)
    return session.exec(query).all()


def get_sub_docs(session: Session, parent_id: int):
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.parent_id == parent_id)

    return session.exec(query).all()


def get(session: Session, doc_id: int):
    query: Select = select(ToolHelpDoc).where(ToolHelpDoc.id == doc_id)

    return session.exec(query).one()
