from collections.abc import Sequence

from sqlalchemy import and_, func, or_, select, true
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import SQLColumnExpression

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


def _visibility_condition(
    user_id: int,
    *,
    include_private: bool,
    include_owned: bool,
) -> SQLColumnExpression[bool]:
    if include_private:
        return true()
    if include_owned:
        return or_(Workflow.public, Workflow.owner_id == user_id)
    return Workflow.public


def to_simple(workflow: Workflow) -> dict:
    """Return the list/search projection for one workflow."""
    interface = workflow.workflow.interface
    return {
        'uid': workflow.uid,
        'name': workflow.name,
        'description': workflow.description,
        'execution_scope': workflow.execution_scope,
        'wf_type': workflow.wf_type,
        'inputs': [{'id': port.id, 'name': port.name} for port in interface.inputs] if interface is not None else [],
        'outputs': [{'id': port.id, 'name': port.name} for port in interface.outputs] if interface is not None else [],
    }


def get_all_by_uid(
    session: Session,
    uids: Sequence[UUIDLike],
    user_id: int,
    *,
    include_private: bool = False,
    include_owned: bool = True,
    wf_type: WorkflowType | None = None,
) -> list[Workflow]:
    """Return accessible workflows in the same relevance order as the requested UIDs."""
    if not uids:
        return []
    normalized_uids = list(dict.fromkeys(as_uuid(uid) for uid in uids))
    condition = _visibility_condition(
        user_id,
        include_private=include_private,
        include_owned=include_owned,
    )
    if wf_type is not None:
        condition = and_(condition, Workflow.wf_type == wf_type)
    statement = select(Workflow).where(condition, Workflow.uid.in_(normalized_uids))
    workflows = {workflow.uid: workflow for workflow in session.scalars(statement).all()}
    return [workflows[uid] for uid in normalized_uids if uid in workflows]


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
    condition = _visibility_condition(
        user_id,
        include_private=include_private,
        include_owned=include_owned,
    )
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
