import asyncio
import json

import polars as pl
from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger
from sqlalchemy import text
from sqlmodel import Session
from tqdm import tqdm

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.crud.tools.tool_group as group_crud
import app.crud.tools.tool_tag as tag_crud
from app.core.config import get_settings
from app.core.session import engine
from app.models.tools import ImageConfig, ToolDefine, ToolGroup, ToolHelpDoc, ToolImage, ToolTag
from llm.models import qwen3_embedding


def init_tag_and_group(drop_old: bool = True):
    with Session(engine) as session:
        if drop_old:
            delete_tag_count = tag_crud.drop_all(session)
            delete_group_count = group_crud.drop_all(session)
            logger.info(f'Delete {delete_tag_count} tags and {delete_group_count} groups')

        # init tags
        with open('../presets/tags.json', 'r') as f:
            tags_data = json.loads(f.read())
            tags = [ToolTag.model_validate(tag) for tag in tags_data]
        tag_crud.insert_all(session, tags)

        # init group
        with open('../presets/groups.json', 'r') as f:
            groups_data = json.loads(f.read())
            groups = [ToolGroup.model_validate(group) for group in groups_data]
        group_crud.insert_all(session, groups)


async def init_images(drop_old: bool = True):
    with Session(engine) as session:
        if drop_old:
            delete_count = image_crud.drop_all(session)
            logger.info(f'Delete {delete_count} data')

        image_df = pl.read_parquet('../presets/images.parquet')
        images: list[ToolImage] = []
        for row in image_df.iter_rows(named=True):
            try:
                image = ImageConfig.from_str(row['image_url'])
                image_data = row
                image_data['image'] = image
                images.append(ToolImage.model_validate(image_data))
            except Exception as e:
                logger.warning(f'Invalid image URL: {row["image_url"]}, {e}')

        images = image_crud.insert_all(session, images)

    embedding = qwen3_embedding()
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name='ImageCollection',
        connection_args={'uri': get_settings().persistent.MILVUS_URL},
        index_params={'index_type': 'HNSW', 'metric_type': 'L2'},
        search_params={'M': 64, 'efConstruction': 100},
        consistency_level='Strong',
        drop_old=drop_old,
        auto_id=True,
    )

    batch_size = 100

    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i : i + batch_size]
        docs = [
            Document(page_content=img.description, metadata={'image_uid': str(img.uid), 'name': img.name})
            for img in batch_images
        ]

        await vectorstore.aadd_documents(docs)


def init_db():
    with Session(engine) as session:
        # init doc
        with open('../presets/tool_docs.json', 'r') as f:
            docs_data = json.loads(f.read())
            docs = [ToolHelpDoc.model_validate(doc) for doc in docs_data]
        doc_crud.insert_all(session, docs)

        # init tool
        with open('../presets/tools.json', 'r') as f:
            tools_data = json.loads(f.read())
        tools = []
        with session.no_autoflush:
            for tool_data in tools_data:
                tool = ToolDefine.model_validate(tool_data)
                tag_ids = [tag['id'] for tag in tool_data['tags']]
                tool.tags = tag_crud.get_all_by_id(session, tag_ids)
                tools.append(tool)
            tool_crud.insert_all(session, tools)

    # reset max_id
    table_seq_pairs = [
        ('tool_tags', 'tool_tags_id_seq'),
        ('tool_groups', 'tool_groups_id_seq'),
        ('images', 'images_id_seq'),
        ('tool_docs', 'tool_docs_id_seq'),
        ('tools', 'tools_id_seq'),
    ]
    with engine.connect() as conn:
        for table, seq in table_seq_pairs:
            max_id = conn.execute(text(f'SELECT MAX(id) FROM {table}')).scalar() or 0
            conn.execute(text(f"SELECT setval('{seq}', :max_id)"), {'max_id': max_id})
        conn.commit()


if __name__ == '__main__':
    init_tag_and_group(True)
    # asyncio.run(init_images())
