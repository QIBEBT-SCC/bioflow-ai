"""Application services for workflow-template persistence and indexing."""

from collections.abc import Sequence

from langchain_core.documents import Document
from sqlalchemy.orm import Session

import app.crud.workflow as workflow_crud
from app.core.config import get_settings
from app.models.workflow import Workflow
from app.schemas.workflow import WorkflowUpdate
from app.services._transaction import committing
from app.utils.identifiers import UUIDLike
from llm.models import qwen3_embedding
from llm.vectorstores import MilvusVectorStore

_WORKFLOW_NOT_FOUND = 'Workflow not found'


class WorkflowCatalogError(Exception):
    """Base exception for workflow-catalog operations."""


class WorkflowCatalogNotFoundError(WorkflowCatalogError):
    """A requested workflow template does not exist."""


async def create_workflow(session: Session, workflow: Workflow) -> Workflow:
    """Persist and index one workflow template."""
    with committing(session):
        workflow_crud.insert(session, workflow)
        await _index_workflow(workflow)
    session.refresh(workflow)
    return workflow


async def create_workflows(session: Session, workflows: Sequence[Workflow]) -> list[Workflow]:
    """Persist and index a batch before committing the database transaction."""
    with committing(session):
        saved_workflows = [workflow_crud.insert(session, workflow) for workflow in workflows]
        for workflow in saved_workflows:
            await _index_workflow(workflow)
    for workflow in saved_workflows:
        session.refresh(workflow)
    return saved_workflows


async def update_workflow(session: Session, uid: UUIDLike, data: WorkflowUpdate) -> Workflow:
    """Update a workflow template in one database transaction."""
    with committing(session):
        workflow = workflow_crud.update(session, uid, data)
        if workflow is None:
            raise WorkflowCatalogNotFoundError(_WORKFLOW_NOT_FOUND)
        await _index_workflow(workflow, replace=True)
    session.refresh(workflow)
    return workflow


async def delete_workflow(session: Session, uid: UUIDLike) -> None:
    """Delete a workflow and its vector projection before committing."""
    with committing(session):
        workflow = workflow_crud.get(session, uid)
        if workflow is None:
            raise WorkflowCatalogNotFoundError(_WORKFLOW_NOT_FOUND)
        workflow_crud.delete(session, uid)
        await _delete_workflow_index(str(workflow.uid))


async def _index_workflow(workflow: Workflow, *, replace: bool = False) -> None:
    """Write the description before removing an existing projection."""
    vectorstore = MilvusVectorStore(
        collection_name='WorkflowCollection',
        embedding_function=qwen3_embedding(),
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    document = Document(
        page_content=workflow.description,
        metadata={
            'wf_uid': str(workflow.uid),
            'name': workflow.name,
            'wf_type': int(workflow.wf_type),
            'owner_id': workflow.owner_id,
            'public': workflow.public,
        },
    )
    async with vectorstore:
        ids = await vectorstore.aadd_documents([document])
        if len(ids) != 1:
            message = 'Workflow vector insertion did not return exactly one ID'
            raise ValueError(message)
        if replace:
            new_ids = [int(value) for value in ids]
            try:
                await vectorstore.adelete(
                    filter_expression='wf_uid == {uid} and pk not in {ids}',
                    filter_params={'uid': str(workflow.uid), 'ids': new_ids},
                )
            except Exception:
                # Do not retain a duplicate if replacing the old projection fails.
                await vectorstore.adelete(
                    filter_expression='wf_uid == {uid} and pk in {ids}',
                    filter_params={'uid': str(workflow.uid), 'ids': new_ids},
                )
                raise


async def _delete_workflow_index(uid: str) -> None:
    """Remove the vector projection before committing the SQL deletion."""
    vectorstore = MilvusVectorStore(
        collection_name='WorkflowCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    async with vectorstore:
        await vectorstore.adelete(filter_expression='wf_uid == {uid}', filter_params={'uid': uid})
