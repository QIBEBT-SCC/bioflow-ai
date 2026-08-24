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
    session.refresh(workflow)
    # await _index_workflows([workflow])
    return workflow


async def create_workflows(session: Session, workflows: Sequence[Workflow]) -> list[Workflow]:
    """Persist a batch of workflow templates in one transaction, then index them."""
    with committing(session):
        saved_workflows = [workflow_crud.insert(session, workflow) for workflow in workflows]
    for workflow in saved_workflows:
        session.refresh(workflow)
    # await _index_workflows(saved_workflows)
    return saved_workflows


def update_workflow(session: Session, uid: UUIDLike, data: WorkflowUpdate) -> Workflow:
    """Update a workflow template in one database transaction."""
    with committing(session):
        workflow = workflow_crud.update(session, uid, data)
        if workflow is None:
            raise WorkflowCatalogNotFoundError(_WORKFLOW_NOT_FOUND)
    session.refresh(workflow)
    return workflow


async def delete_workflow(session: Session, uid: UUIDLike) -> None:
    """Delete a workflow template, then remove its vector projection."""
    with committing(session):
        if not workflow_crud.delete(session, uid):
            raise WorkflowCatalogNotFoundError(_WORKFLOW_NOT_FOUND)

    vectorstore = MilvusVectorStore(
        collection_name='WorkflowCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    await vectorstore.adelete(
        filter_expression='wf_uid == {uid}',
        filter_params={'uid': str(uid)},
    )


async def _index_workflows(workflows: Sequence[Workflow]) -> None:
    """Add workflow description projections to the vector index."""
    if not workflows:
        return

    vectorstore = MilvusVectorStore(
        collection_name='WorkflowCollection',
        embedding_function=qwen3_embedding(),
        uri=get_settings().knowledge_base.MILVUS_URL,
        hybrid=True,
    )
    documents = [
        Document(
            page_content=workflow.description,
            metadata={'wf_uid': str(workflow.uid), 'name': workflow.name},
        )
        for workflow in workflows
    ]
    async with vectorstore:
        await vectorstore.aadd_documents(documents)
