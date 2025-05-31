from enum import IntEnum
from typing import Optional, Any

from pydantic import UUID7, BaseModel, Field as PField
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import SQLModel, Field, Relationship

from app.models.decorator.special_field import PydanticJson


class ToolType(IntEnum):
    DOCKER = 0
    COMMAND_LINE = 1


class ParamType(IntEnum):
    INPUT = 0
    INPUT_POSITION = 1
    OUTPUT = 2


class ParamDefine(BaseModel):
    """命令行参数定义"""

    name: str = PField(description='参数的名称，用于在模板字符串中作为格式化关键词，只允许小写字母和下划线')
    command: str = PField(description='具体的参数定义，使用name字段的值构建的模板字符串。如 "-i {name}"')
    param_type: ParamType = PField(description='参数的类型，0代表INPUT、1代表INPUT_POSITION、2代表OUTPUT')
    is_file: bool = PField(description='参数涉及的是文件还是文件夹')
    index: Optional[int] = PField(description='位置参数的顺位（仅在INPUT_POSITION类型的参数中生效）', default=None)
    required: bool = PField(description='是否是程序运行的必须参数', default=True)
    mount_path: Optional[str] = PField(description='在docker中的映射目录', default=None)
    description: Optional[str] = PField(description='参数描述', default=None)


class OutputFile(BaseModel):
    """输出文件定义"""

    name: str = PField(description='文件参数名称，只允许小写字母和下划线')
    file_path: str = PField(description='文件的路径模板，只允许包含output_dir和sample_name这两个变量名')
    is_report: bool = PField(description='是否作为report文件', default=False)
    is_log: bool = PField(description='是否是日志文件', default=False)
    mount_path: Optional[str] = PField(description='文件在容器中的映射路径', default=None)
    description: Optional[str] = PField(description='文件的描述', default=None)


class ImageConfig(SQLModel):
    repository: str
    tag: str = 'latest'

    def get_image(self) -> str:
        return f'{self.repository}:{self.tag}'


class DockerConfig(SQLModel):
    image: ImageConfig
    docker_settings: dict[str, Any]


class ToolGroup(SQLModel, table=True):
    __tablename__ = 'tool_groups'

    id: int = Field(primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')


class ToolTagLink(SQLModel, table=True):
    __tablename__ = 'tool_tag_link'

    tool_id: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('tools.uid')))
    tag_id: int = Field(foreign_key='tool_tags.id', primary_key=True)


class ToolTag(SQLModel, table=True):
    __tablename__ = 'tool_tags'
    model_config = {'ignored_types': (hybrid_property,)}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)

    tools: list['ToolData'] = Relationship(back_populates='tags', link_model=ToolTagLink)


class ToolData(SQLModel, table=True):
    __tablename__ = 'tools'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    name: str
    description: str
    homepage: str
    tool_type: ToolType
    group_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')

    command_template: str
    dynamic_params: list[ParamDefine] = Field(sa_column=Column(PydanticJson(list[ParamDefine])))
    static_params: str
    output_files: list[OutputFile] = Field(sa_column=Column(PydanticJson(list[OutputFile])))

    help_command: str

    mkdir_output: bool
    use_temp_dir: bool

    docker_config: Optional[DockerConfig] = Field(sa_column=Column(PydanticJson(DockerConfig)))

    tags: list['ToolTag'] = Relationship(back_populates='tools', link_model=ToolTagLink)
