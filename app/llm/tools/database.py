from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.image as image_crud
import app.crud.tools.tool as tool_crud
import app.crud.resource.bio_db as db_crud
from app.core.session import engine
from app.llm.formatters import handle_formatter
from app.models.resource import BioDb
from app.models.tools import ToolImage, ToolDefine
from app.schemas.tools import ToolNodePublic


class SearchLocalImageSchema(BaseModel):
    query: str = Field(description='Search keywords, generally tool names')


class SearchLocalImage(BaseTool):
    name: str = 'search_local_image'
    description: str = 'Search the database for images that have already been pulled locally.'
    args_schema: Optional[ArgsSchema] = SearchLocalImageSchema
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        with Session(engine) as session:
            images: list[ToolImage] = image_crud.get_by_name_fuzzy(session, '', 50, 0)

        if images:
            image_blocks = [
                (
                    f'  <image>\n'
                    f'    <id>{image.id}</id>\n'
                    f'    <name>{image.name}</name>\n'
                    f'    <description>{image.description}</description>\n'
                    f'    <url>{image.image.get_image()}</url>\n'
                    f'  <image>'
                )
                for image in images
            ]
            return f'Based on the keywords you provided, the following potentially related images were found:\n<images>\n{"\n\n".join(image_blocks)}\n</images>'

        return 'Based on the keywords you provided, no results found'


class SearchToolSchema(BaseModel):
    query: str = Field(description='Tool name, used to search for corresponding tool nodes')


class SearchTool(BaseTool):
    name: str = 'search_tool'
    description: str = 'Search for the corresponding tool node'
    args_schema: Optional[ArgsSchema] = SearchToolSchema
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        with Session(engine) as session:
            exist_tools: list[ToolDefine] = tool_crud.get_by_name_fuzzy(session, query, 50, 0)

        if exist_tools:
            tool_nodes = [ToolNodePublic.model_validate(exist_tool.model_dump()) for exist_tool in exist_tools]
            tool_blocks = [
                (
                    f'  <tool>\n'
                    f'    <name>{t.name}</name>\n'
                    f'    <description>{t.description}</description>\n'
                    f'    <input_handles>\n'
                    f'{handle_formatter(t.input_handles)}\n'
                    f'    </input_handles>\n'
                    f'    <output_handles>\n'
                    f'{handle_formatter(t.output_handles)}\n'
                    f'    </output_handles>\n'
                    f'    <data>\n'
                    f'      <tool_id>{t.id}</tool_id>\n'
                    f'      <args>{t.static_params}<args>\n'
                    f'    </data>\n'
                    f'    <command>{t.complete_command}</command>'
                    f'  </tool>'
                )
                for t in tool_nodes
            ]
            return f'Based on the keywords, the following tool nodes are available (To aid in judgment, each tool node has an additional <command> field, indicating the command actually executed by that node):\n<tools>\n{"\n\n".join(tool_blocks)}\n</tools>'

        return 'Based on the keywords you provided, no results found'


class SearchBioDBSchema(BaseModel):
    query: str = Field(description='Keyword for Search')


class SearchBioDB(BaseTool):
    name: str = 'search_biodb'
    description: str = 'Search for the biological databases used by the tool using its name'
    args_schema: Optional[ArgsSchema] = SearchBioDBSchema
    return_direct: bool = False

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        with Session(engine) as session:
            exist_dbs: list[BioDb] = db_crud.get_by_name_fuzzy(session, query, 10, offset=0)

        if exist_dbs:
            db_blocks = [
                (
                    f'<db>\n'
                    f'  <id>{db.id}</id>\n'
                    f'  <name>{db.name}</name>\n'
                    f'  <description>{db.description}</description>\n'
                    f'</db>'
                )
                for db in exist_dbs
            ]

            return f'Based on the keywords, the following db nodes are available:\n\n{"\n".join(db_blocks)}'

        return 'Based on the keywords you provided, no results found'


search_local_image = SearchLocalImage()
search_tool = SearchTool()
search_biodb = SearchBioDB()