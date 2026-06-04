from sqlalchemy import func
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
    statement = select(Workflow).where(Workflow.name == name)
    result = session.exec(statement).first()
    return result is not None


def get_page(session: Session, limit: int, offset: int, user_id: int) -> tuple[list[dict], int]:
    """Return a page of workflows accessible to a user together with the total count."""
    condition = or_(Workflow.public, Workflow.owner_id == user_id)
    page_stmt = (
        select(Workflow.uid, Workflow.name, Workflow.description, Workflow.execution_scope)
        .where(condition)
        .offset(offset)
        .limit(limit)
    )
    count_stmt = select(func.count()).select_from(Workflow).where(condition)

    result = session.exec(page_stmt).all()
    data = [
        {'uid': uid, 'name': name, 'description': description, 'execution_scope': execution_scope}
        for uid, name, description, execution_scope in result
    ]
    return data, session.exec(count_stmt).one()


def get(session: Session, wf_uid: str) -> Workflow:
    """Return a workflow by its uid."""
    statement = select(Workflow).where(Workflow.uid == wf_uid)
    return session.exec(statement).one()
