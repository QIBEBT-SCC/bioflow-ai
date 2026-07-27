"""Application services for reusable workflow CODE definitions."""

from langchain_core.documents import Document
from sqlalchemy.orm import Session

import app.crud.code as code_crud
from app.core.config import get_settings
from app.models.code import CodeDefine
from app.schemas.code import CodeCreate
from app.services._transaction import committing
from llm.models import qwen3_embedding
from llm.vectorstores import MilvusVectorStore

AUXILIARY_NODE_COLLECTION = 'ProcessorCollection'
CODE_NODE_KIND = 'code'
PROCESSOR_NODE_KIND = 'processor'


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
