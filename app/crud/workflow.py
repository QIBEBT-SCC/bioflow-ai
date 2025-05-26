from sqlalchemy import Select, func
from sqlmodel import Session, select, or_

from app.models.react_flow import WorkflowDefinition
from app.models.workflow import Workflow


def insert(session: Session, workflow: Workflow) -> Workflow:
    session.add(workflow)
    session.commit()
    session.refresh(workflow)

    return workflow


def update(session: Session, uid: str, workflow: WorkflowDefinition) -> Workflow:
    db_flow = get(session, uid)

    db_flow.workflow = workflow
    session.add(db_flow)
    session.commit()
    session.refresh(db_flow)

    return db_flow


def workflow_name_exists(session: Session, name: str) -> bool:
    statement: Select = select(Workflow).where(Workflow.name == name)
    result = session.exec(statement).first()
    return result is not None


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
