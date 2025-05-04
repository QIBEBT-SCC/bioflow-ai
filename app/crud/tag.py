from typing import Sequence

from sqlmodel import Session, select

from app.models.project import Tag


def insert(session: Session, tag: Tag) -> Tag:
    session.add(tag)
    session.commit()
    session.refresh(tag)

    return tag


def get_all(session: Session) -> Sequence[Tag]:
    statement = select(Tag)
    return session.exec(statement).all()
