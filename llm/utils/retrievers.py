from typing import TYPE_CHECKING
from uuid import UUID

from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sqlmodel import Session, col, select

import app.crud.tools.tool as tool_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.document import ParentDocument
from app.schemas.tools import ToolNodePublic
from llm.models import qwen3_embedding
from llm.utils.uid_mapper import BidirectionalMapper

if TYPE_CHECKING:
    from app.models.tools import ToolDefine


def insert_documents(document_content: str, doc_id: str) -> None:
    """插入文档: 父文档存数据库, 子文档存 Milvus."""
    embedding_model = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='PaperCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding_model,
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        consistency_level='Strong',
        drop_old=False,
        auto_id=True,
    )

    # 1. 分割父文档 (按 Markdown 标题)
    headers_to_split_on = [('#', 'Title'), ('##', 'Section')]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    parent_docs = parent_splitter.split_text(document_content)

    # 2. 子文档分割器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    with Session(engine) as session:
        for parent_doc in parent_docs:
            # 3. 保存父文档到数据库
            parent = ParentDocument(
                doc_id=doc_id,
                content=parent_doc.page_content,
                title=parent_doc.metadata.get('Title', ''),
                section=parent_doc.metadata.get('Section', ''),
            )
            session.add(parent)
            session.commit()
            session.refresh(parent)

            # 4. 分割子文档
            child_docs = child_splitter.split_documents([parent_doc])

            # 5. 子文档添加 parent_uid 元数据
            for child_doc in child_docs:
                child_doc.metadata['doc_id'] = doc_id
                child_doc.metadata['parent_uid'] = str(parent.uid)
                child_doc.metadata['Title'] = parent.title
                child_doc.metadata['Section'] = parent.section

            # 6. 子文档分批存到 Milvus
            batch_size = get_settings().pdf_parser.BATCH_SIZE
            for i in range(0, len(child_docs), batch_size):
                batch = child_docs[i : i + batch_size]
                vectorstore.add_documents(batch)


async def retrieve_documents(query: str, doc_id: str | None = None, k: int = 8) -> list[Document]:
    """从 Milvus 查子文档, 根据 parent_uid 从数据库取父文档."""
    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='Papers',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
    )

    task = 'Provide a search query to search for relevant paper snippets'
    query = f'Instruct: {task}\nQuery:{query}'

    # 1. 从 Milvus 查询子文档
    if doc_id:
        results = await vectorstore.amax_marginal_relevance_search(query, k=k, expr=f'doc_id == "{doc_id}"')
    else:
        results = await vectorstore.amax_marginal_relevance_search(query, k=k)

    if not results:
        return []

    # 2. 提取 parent_uid 并去重
    parent_uids = list({UUID(r.metadata['parent_uid']) for r in results})

    # 3. 从数据库批量查询父文档
    with Session(engine) as session:
        statement = select(ParentDocument).where(col(ParentDocument.uid).in_(parent_uids))
        parent_docs = session.exec(statement).all()

    # 4. 转换为 Document 对象
    return [
        Document(
            page_content=p.content,
            metadata={
                'doc_id': p.doc_id,
                'parent_uid': str(p.uid),
                'Title': p.title,
                'Section': p.section,
            },
        )
        for p in parent_docs
    ]


async def retrieve_tools(
    query: str, image_uid: UUID | None = None
) -> tuple[list[ToolNodePublic], BidirectionalMapper | None]:
    """Search existing tools from the vector database based on functionality."""
    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ToolCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        builtin_function=BM25BuiltInFunction(),
        vector_field=['dense', 'sparse'],
        consistency_level='Strong',
    )

    task = 'Provide a search query to search for tools with relevant functions'
    query = f'Instruct: {task}\nQuery:{query}'

    if image_uid:
        results = await vectorstore.asimilarity_search(
            query,
            k=10,
            param=[{'M': 64, 'efConstruction': 100}, {}],
            expr=f'image_uid == "{image_uid!s}"',
            ranker_type='weighted',
            ranker_params={'weights': [0.85, 0.15]},
        )
    else:
        results = await vectorstore.asimilarity_search(
            query,
            k=10,
            param=[{'M': 64, 'efConstruction': 100}, {}],
            ranker_type='weighted',
            ranker_params={'weights': [0.85, 0.15]},
        )

    if not results:
        return [], None

    with Session(engine) as session:
        exist_tools: list[ToolDefine] = tool_crud.get_all_by_uid(session, [t.metadata['tool_uid'] for t in results])
        mappers = BidirectionalMapper(datas=exist_tools)

    tool_nodes = [ToolNodePublic.model_validate(exist_tool.model_dump()) for exist_tool in exist_tools]

    return tool_nodes, mappers


async def retrieve_processor(query: str) -> list[Document]:
    """Search for the processor node corresponding to the function."""
    embedding = qwen3_embedding()
    vectorstore = Milvus(
        collection_name='ProcessorCollection',
        connection_args={'uri': get_settings().knowledge_base.MILVUS_URL},
        embedding_function=embedding,
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
    )

    task = 'Provide a search query to search for data processors with relevant functions'
    query = f'Instruct: {task}\nQuery:{query}'

    return await vectorstore.amax_marginal_relevance_search(query, k=10)
