"""Persistence helpers for workflow-builder drafts."""

import datetime
import uuid

from langchain_core.documents import Document
from sqlmodel import Session

import app.crud.workflow as wf_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.workflow import Workflow, WorkflowType
from llm.models import qwen3_embedding
from llm.states import WorkflowDraft
from llm.vectorstores import MilvusVectorStore
from llm.workflow_builder.serialization import build_workflow_definition, to_execution_scope


async def save_workflow_drafts(drafts: list[WorkflowDraft], user_id: int) -> list[Workflow]:
    """Persist workflow drafts and index their descriptions for retrieval."""
    now = datetime.datetime.now(datetime.UTC)
    saved_workflows = []
    with Session(engine) as session:
        for draft in drafts:
            workflow_define = build_workflow_definition(draft.nodes, draft.edges)
            workflow = Workflow(
                name=f'{now.strftime("%y%m%d")}_{draft.name}(AI)',
                description=draft.description,
                owner_id=user_id,
                wf_type=WorkflowType.TEMPLATE,
                execution_scope=to_execution_scope(draft.execution_scope),
                workflow=workflow_define,
                public=True,
                auto_summary=True,
            )
            saved_workflows.append(wf_crud.insert(session, workflow))

        embedding_model = qwen3_embedding()
        vectorstore = MilvusVectorStore(
            collection_name='WorkflowCollection',
            embedding_function=embedding_model,
            uri=get_settings().knowledge_base.MILVUS_URL,
            hybrid=True,
        )

        docs = [
            Document(
                page_content=workflow.description,
                metadata={
                    'wf_uid': str(workflow.uid),
                    'name': workflow.name,
                },
            )
            for workflow in saved_workflows
        ]
        async with vectorstore:
            await vectorstore.aadd_documents(docs)

    return saved_workflows


def workflow_uids(workflows: list[Workflow]) -> list[uuid.UUID]:
    """Return saved workflow UIDs with a concrete UUID type."""
    return [workflow.uid for workflow in workflows]
