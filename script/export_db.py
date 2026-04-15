from typing import TYPE_CHECKING

import polars as pl
from sqlmodel import Session
from tqdm import tqdm

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.tools.tool_doc as doc_crud
import app.models.user  # noqa: F401
from app.core.session import engine

if TYPE_CHECKING:
    from app.models.tools import ToolDefine, ToolHelpDoc, ToolImage


def export_images() -> None:
    """导出镜像数据."""
    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = image_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            images: list[ToolImage] = image_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'uid': str(image.uid),
                    'name': image.name,
                    'version': image.version,
                    'description': image.description,
                    'homepage': image.homepage,
                    'paper_link': image.paper_link,
                    'image_url': image.image.get_image(),
                }
                for image in images
            ]
            all_data.extend(batch_data)

    output_df = pl.DataFrame(all_data)
    output_df.write_parquet('../presets/images.parquet', compression='snappy')


def export_docs() -> None:
    """导出工具文档."""
    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = doc_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            docs: list[ToolHelpDoc] = doc_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'id': str(doc.uid),
                    'help_command': doc.help_command,
                    'description': doc.description if doc.description else '',
                    'content': doc.content,
                    'image_uid': str(doc.image_uid),
                }
                for doc in docs
            ]
            all_data.extend(batch_data)

    output_df = pl.DataFrame(all_data)
    output_df.write_parquet('../presets/tool_docs.parquet', compression='snappy')


def export_tools() -> None:
    """导出工具节点."""
    batch_size = 100
    all_data = []

    with Session(engine) as session:
        total_count = tool_crud.get_counts(session)

        for offset in tqdm(range(0, total_count, batch_size)):
            tools: list[ToolDefine] = tool_crud.get_all(session, batch_size, offset)

            batch_data = [
                {
                    'uid': str(tool.uid),
                    'name': tool.name,
                    'image_uid': str(tool.image_uid),
                    'description': tool.description,
                    'tool_type': tool.tool_type,
                    'group_id': tool.group_id,
                    'help_doc_uid': str(tool.help_doc_uid),
                    'command_template': tool.command_template,
                    'dynamic_params': [param.model_dump() for param in tool.dynamic_params],
                    'immutable_static_params': tool.immutable_static_params,
                    'modifiable_static_params': tool.modifiable_static_params,
                    'file_mounts': [output.model_dump() for output in tool.file_mounts],
                    'tags': [tag.model_dump() for tag in tool.tags],
                }
                for tool in tools
            ]
            all_data.extend(batch_data)

        if all_data:
            output_df = pl.DataFrame(all_data)
            output_df.write_parquet('../presets/tools.parquet', compression='snappy')


if __name__ == '__main__':
    export_images()
    export_docs()
    export_tools()
