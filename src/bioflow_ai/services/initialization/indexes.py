from anyio import Path
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioflow_ai.services.progress import ProgressIterator, with_progress


async def rebuild_image_index(*, progress: ProgressIterator | None = None) -> None:
    """重建镜像向量索引."""
    from langchain_core.documents import Document

    from bioflow_ai.core.config import get_settings
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.llm.models import qwen3_embedding
    from bioflow_ai.llm.vectorstores import MilvusVectorStore
    from bioflow_ai.models.tools import ToolImage

    embedding = qwen3_embedding()
    vectorstore = MilvusVectorStore(
        embedding_function=embedding,
        collection_name='ImageCollection',
        uri=get_settings().knowledge_base.MILVUS_URL,
        drop_old=True,
    )

    with Session(engine) as session:
        statement = select(
            ToolImage.uid,
            ToolImage.name,
            ToolImage.description,
        )
        results = session.execute(statement).all()

    images_to_add = []
    empty_images = []
    batch_size = 10

    for image_data in with_progress(results, progress):
        if image_data.description:
            images_to_add.append(
                Document(
                    page_content=image_data.description,
                    metadata={
                        'image_uid': str(image_data.uid),
                        'name': image_data.name,
                    },
                )
            )

            if len(images_to_add) >= batch_size:
                async with vectorstore:
                    await vectorstore.aadd_documents(images_to_add)
                images_to_add = []
        else:
            empty_images.append(str(image_data.uid))

    if images_to_add:
        async with vectorstore:
            await vectorstore.aadd_documents(images_to_add)

    if empty_images:
        empty_file = Path('empty_images.txt')
        await empty_file.write_text('\n'.join(empty_images) + '\n')


async def rebuild_tool_index(*, progress: ProgressIterator | None = None) -> None:
    """重建工具库向量索引."""
    from bioflow_ai.infrastructure.database import engine
    from bioflow_ai.models.tools import ToolDefine
    from bioflow_ai.services.tools.index import index_tools

    with Session(engine) as session:
        tools = session.scalars(select(ToolDefine)).all()

    empty_tools = await index_tools(tools, drop_old=True, progress=progress)

    if empty_tools:
        empty_file = Path('empty_tools.txt')
        await empty_file.write_text('\n'.join(str(tool.uid) for tool in empty_tools) + '\n')
