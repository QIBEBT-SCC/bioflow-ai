"""Application services for reusable workflow CODE definitions."""

from langchain_core.documents import Document
from sqlalchemy.orm import Session

import bioflow_ai.crud.code as code_crud
from bioflow_ai.core.config import get_settings
from bioflow_ai.llm.models import qwen3_embedding
from bioflow_ai.llm.vectorstores import MilvusVectorStore
from bioflow_ai.models.code import CodeDefine
from bioflow_ai.schemas.code import CodeCreate, CodeUpdate
from bioflow_ai.services._transaction import committing

AUXILIARY_NODE_COLLECTION = 'ProcessorCollection'
CODE_NODE_KIND = 'code'
PROCESSOR_NODE_KIND = 'processor'


class CodeCatalogValidationError(ValueError):
    """Raised when a CODE catalog mutation violates domain constraints."""


def code_index_document(code: CodeDefine) -> Document:
    """Build the vector document used to discover a reusable CODE definition."""
    return Document(
        page_content=code.description,
        metadata={
            'kind': CODE_NODE_KIND,
            'code_uid': str(code.uid),
            'name': code.name,
            'node_type': code.node_type,
        },
    )


async def index_code(code: CodeDefine) -> None:
    """Add one reusable CODE definition to the shared auxiliary-node index."""
    vectorstore = MilvusVectorStore(
        collection_name=AUXILIARY_NODE_COLLECTION,
        embedding_function=qwen3_embedding(),
        uri=get_settings().knowledge_base.MILVUS_URL,
    )
    async with vectorstore:
        await vectorstore.aadd_documents([code_index_document(code)])


async def _delete_code_index(code_uid: str) -> None:
    """Delete one CODE definition projection from the shared auxiliary index."""
    vectorstore = MilvusVectorStore(
        collection_name=AUXILIARY_NODE_COLLECTION,
        uri=get_settings().knowledge_base.MILVUS_URL,
    )
    async with vectorstore:
        await vectorstore.adelete(
            filter_expression='code_uid == {uid}',
            filter_params={'uid': code_uid},
        )


async def _refresh_code_index(code: CodeDefine) -> None:
    """Replace one CODE definition projection after searchable metadata changes."""
    await _delete_code_index(str(code.uid))
    await index_code(code)


async def create_code(session: Session, data: CodeCreate) -> CodeDefine:
    """Persist and index one reusable CODE definition."""
    code = CodeDefine(**data.model_dump())
    with committing(session):
        code_crud.insert(session, code)

    try:
        await index_code(code)
    except Exception:
        with committing(session):
            code_crud.delete(session, code.uid)
        raise

    return code


async def update_code(session: Session, code_uid: str, data: CodeUpdate) -> CodeDefine | None:
    """Update an existing CODE definition while preserving its node type."""
    code = code_crud.get(session, code_uid)
    if code is None:
        return None

    vector_projection_changed = ('name' in data.model_fields_set and data.name != code.name) or (
        'description' in data.model_fields_set and data.description != code.description
    )

    with committing(session):
        updated = code_crud.update(session, code_uid, data)

    if updated is not None and vector_projection_changed:
        await _refresh_code_index(updated)
    return updated


async def delete_code(session: Session, code_uid: str) -> bool:
    """Delete a CODE definition and its vector projection."""
    with committing(session):
        deleted = code_crud.delete(session, code_uid)
    if not deleted:
        return False

    await _delete_code_index(str(code_uid))
    return True
