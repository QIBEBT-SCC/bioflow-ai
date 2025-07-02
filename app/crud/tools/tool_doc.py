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


def get_by_name_fuzzy(session: Session, command: str, limit: int, offset: int):
    statement: Select = (
        select(ToolHelpDoc)
        .where(col(ToolHelpDoc.help_command).ilike(f'%{command}%'))
        .where(col(ToolHelpDoc.parent_id).is_(None))
        .offset(offset)
        .limit(limit)
    )

    return session.exec(statement).all()


def get_sub_docs(session: Session, parent_id: int):
    statement: Select = select(ToolHelpDoc).where(ToolHelpDoc.parent_id == parent_id)

    return session.exec(statement).all()


def get(session: Session, doc_id: int):
    statement: Select = select(ToolHelpDoc).where(ToolHelpDoc.id == doc_id)

    return session.exec(statement).one()
