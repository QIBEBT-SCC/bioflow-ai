from sqlalchemy import Select, func
from sqlmodel import Session, or_, select

from app.models.workflow import Workflow
from app.schemas.workflow import WorkflowUpdate


def insert(session: Session, workflow: Workflow) -> Workflow:
    """Insert a new workflow into the database."""
    session.add(workflow)
    session.commit()
    session.refresh(workflow)

    return workflow


def update(session: Session, uid: str, data: WorkflowUpdate) -> Workflow:
    """Update an existing workflow with new data."""
    db_flow = get(session, uid)

    if data.name is not None:
        db_flow.name = data.name
    if data.workflow is not None:
        db_flow.workflow = data.workflow
    session.add(db_flow)
    session.commit()
    session.refresh(db_flow)

    return db_flow


def delete(session: Session, uid: str) -> None:
    """Delete a workflow by its uid."""
    db_flow = get(session, uid)
    session.delete(db_flow)
    session.commit()


def workflow_name_exists(session: Session, name: str) -> bool:
    """Return True if a workflow with the given name already exists."""
    statement: Select = select(Workflow).where(Workflow.name == name)
    result = session.exec(statement).first()
    return result is not None


def get_all(session: Session, limit: int, offset: int, user_id: int) -> list[dict]:
    """Return a paginated list of workflows accessible to a user."""
    statement: Select = (
        select(Workflow.uid, Workflow.name)
        .where(or_(Workflow.public, Workflow.owner_id == user_id))
        .offset(offset)
        .limit(limit)
    )

    result = session.exec(statement).all()
    return [{'uid': uid, 'name': name} for uid, name in result]


def get_count(session: Session, user_id: int) -> int:
    """Return the total count of workflows accessible to a user."""
    statement: Select = (
        select(func.count())
        .select_from(Workflow)
        .where(or_(Workflow.public, Workflow.owner_id == user_id))
    )

    return session.exec(statement).one()


def get(session: Session, wf_uid: str) -> Workflow:
    """Return a workflow by its uid."""
    statement: Select = select(Workflow).where(Workflow.uid == wf_uid)
    return session.exec(statement).one()
