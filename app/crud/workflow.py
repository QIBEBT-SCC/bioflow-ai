from sqlalchemy import and_, func, or_, select, true
from sqlalchemy.orm import Session

from app.models.workflow import Workflow
from app.schemas.workflow import WorkflowUpdate
from app.schemas.workflow_config import WorkflowType
from app.utils.identifiers import UUIDLike, as_uuid


def insert(session: Session, workflow: Workflow) -> Workflow:
    """Insert a new workflow into the database."""
    session.add(workflow)
    session.flush()
    session.refresh(workflow)
    return workflow


def update(session: Session, uid: UUIDLike, data: WorkflowUpdate) -> Workflow | None:
    """Update an existing workflow with new data."""
    db_flow = get(session, uid)
    if db_flow is None:
        return None

    if data.name is not None:
        db_flow.name = data.name
    if data.workflow is not None:
        db_flow.workflow = data.workflow
    for field in ('description', 'public', 'execution_scope'):
        if getattr(data, field) is not None:
            setattr(db_flow, field, getattr(data, field))
    session.add(db_flow)
    session.flush()
    session.refresh(db_flow)
    return db_flow


def delete(session: Session, uid: UUIDLike) -> bool:
    """Delete a workflow by its uid."""
    db_flow = get(session, uid)
    if db_flow is None:
        return False
    session.delete(db_flow)
    session.flush()
    return True


def workflow_name_exists(session: Session, name: str) -> bool:
    """Return True if a workflow with the given name already exists."""
    statement = select(Workflow).where(Workflow.name == name)
    return session.scalar(statement) is not None


def get_page(
    session: Session,
    limit: int,
    offset: int,
    user_id: int,
    *,
    include_private: bool = False,
    include_owned: bool = True,
    wf_type: WorkflowType | None = None,
) -> tuple[list[dict], int]:
    """Return a page of workflows accessible to a user together with the total count."""
    if include_private:
        condition = true()
    elif include_owned:
        condition = or_(Workflow.public, Workflow.owner_id == user_id)
    else:
        condition = Workflow.public
    if wf_type is not None:
        condition = and_(condition, Workflow.wf_type == wf_type)
    columns = (
        Workflow.uid,
        Workflow.name,
        Workflow.description,
        Workflow.execution_scope,
        Workflow.wf_type,
    )
    include_interface = wf_type == WorkflowType.SUBMODULE
    page_stmt = select(*columns, *((Workflow.workflow,) if include_interface else ())).where(condition)
    page_stmt = page_stmt.offset(offset).limit(limit)
    count_stmt = select(func.count()).select_from(Workflow).where(condition)

    result = session.execute(page_stmt).all()
    data = []
    for row in result:
        uid, name, description, execution_scope, workflow_type = row[:5]
        interface = row[5].interface if include_interface else None
        data.append(
            {
                'uid': uid,
                'name': name,
                'description': description,
                'execution_scope': execution_scope,
                'wf_type': workflow_type,
                'inputs': (
                    [{'id': port.id, 'name': port.name} for port in interface.inputs] if interface is not None else []
                ),
                'outputs': (
                    [{'id': port.id, 'name': port.name} for port in interface.outputs] if interface is not None else []
                ),
            }
        )
    return data, session.scalar(count_stmt) or 0


def get(session: Session, wf_uid: UUIDLike) -> Workflow | None:
    """Return a workflow by its uid."""
    statement = select(Workflow).where(Workflow.uid == as_uuid(wf_uid))
    return session.scalar(statement)
