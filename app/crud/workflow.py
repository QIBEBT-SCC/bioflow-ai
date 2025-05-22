from sqlalchemy import Select, func
from sqlmodel import Session, select, or_

from app.models.workflow import Workflow


def insert(session: Session, workflow: Workflow) -> Workflow:
    session.add(workflow)
    session.commit()
    session.refresh(workflow)

    return workflow


def get_all(session: Session, limit: int, offset: int, user_id: int):
    statement: Select = (
        select(Workflow.uid, Workflow.name)
        .where(or_(Workflow.public, Workflow.owner_id == user_id))
        .offset(offset)
        .limit(limit)
    )

    result = session.exec(statement).all()
    return [{'uid': uid, 'name': name} for uid, name in result]


def get_count(session: Session, user_id: int) -> int:
    statement: Select = (
        select(func.count())
        .select_from(Workflow)
        .where(or_(Workflow.public, Workflow.owner_id == user_id))
    )

    return session.exec(statement).one()


def get(session: Session, wf_uid: str) -> Workflow:
    statement: Select = select(Workflow).where(Workflow.uid == wf_uid)
    return session.exec(statement).one()
