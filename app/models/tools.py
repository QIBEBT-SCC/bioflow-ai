from enum import IntEnum
from typing import Optional, Any

from pydantic import BaseModel, UUID7
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import SQLModel, Field, Relationship
from uuid_extensions import uuid7

from pydantic import Field as PField

from app.models.decorator.special_field import PydanticJson


class ToolType(IntEnum):
    DOCKER = 0
    COMMAND_LINE = 1


class ParamType(IntEnum):
    INPUT = 0
    INPUT_POSITION = 1
    OUTPUT = 2


class ParamDefine(BaseModel):
    """命令行参数定义

    Attributes:
        name: 参数名称
        command: 参数命令格式，如 "-i {name}"
        param_type: 是输入相关参数还是输出相关参数
        is_file: 输入是文件还是文件夹
        index: 位置参数的顺位（仅在INPUT_POSITION类型的参数中生效）
        required: 是否必选
        mount_path: 在docker中的映射目录
        description: 参数描述
    """

    name: str
    command: str
    param_type: ParamType
    is_file: bool
    index: Optional[int] = None
    required: bool = True
    mount_path: Optional[str] = None
    description: Optional[str] = None


class OutputFile(BaseModel):
    """输出文件定义

    Attributes:
        name: 文件参数名称
        file_path: 文件的路径模板
        is_report: 是否作为report文件
        is_log: 是否是日志文件
        mount_path: 文件在容器中的映射路径
        description: 文件的描述
    """

    name: str
    file_path: str
    is_report: bool = False
    is_log: bool = False
    mount_path: Optional[str] = None
    description: Optional[str] = None


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


class DockerToolCreate(BaseModel):
    """创建Docker工具的请求模型"""

    name: str
    description: str
    tool_tag: list[ToolTag]

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    help_command: str
    homepage: str

    mkdir_output: bool = True
    use_temp_dir: bool = False

    repository: str
    tag: str = 'latest'
    docker_config: dict[str, Any] = {}

    def to_tool_data(self) -> ToolData:
        tool_id = uuid7(as_type='str')
        return ToolData(
            uid=str(tool_id),
            name=self.name,
            description=self.description,
            tags=self.tool_tag,
            tool_type=ToolType.DOCKER,
            command_template=self.command_template,
            homepage=self.homepage,
            dynamic_params=self.dynamic_params,
            static_params=self.static_params,
            output_files=self.output_files,
            help_command=self.help_command,
            mkdir_output=self.mkdir_output,
            use_temp_dir=self.use_temp_dir,
            docker_config=DockerConfig(
                image=ImageConfig(repository=self.repository, tag=self.tag),
                docker_settings=self.docker_config,
            ),
        )


class SimpleToolPublic(BaseModel):
    uid: UUID7
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    docker_image: str

    tags: list[ToolTag]

    @classmethod
    def from_orm(cls, tool: ToolData, update: Optional[dict[str, Any]] = None):
        kwargs = tool.model_dump()

        if tool.tool_type == ToolType.DOCKER:
            kwargs['docker_image'] = tool.docker_config.image.get_image()

        kwargs['tags'] = tool.tags

        if update:
            kwargs.update(update)
        return cls.model_validate(kwargs)


class ToolPublic(BaseModel):
    uid: UUID7
    name: str
    description: str
    homepage: str
    tool_type: ToolType
    group_id: Optional[int]

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    help_command: str
    complete_command: str = ''

    mkdir_output: bool
    use_temp_dir: bool

    docker_repo: Optional[str] = None
    docker_tag: Optional[str] = None

    tags: list['ToolTag']

    @classmethod
    def from_orm(cls, tool: ToolData, update: Optional[dict[str, Any]] = None):
        kwargs = tool.model_dump()

        if tool.tool_type == ToolType.DOCKER:
            kwargs['docker_repo'] = tool.docker_config.image.repository
            kwargs['docker_tag'] = tool.docker_config.image.tag

        kwargs['tags'] = tool.tags

        if update:
            kwargs.update(update)
        return cls.model_validate(kwargs)


class Handle(BaseModel):
    name: str
    description: str


class ToolArgPublic(BaseModel):
    uid: UUID7
    name: str
    description: str

    input_handles: list[Handle] = PField(default_factory=list)
    output_handles: list[Handle] = PField(default_factory=list)

    static_params: str

    @classmethod
    def from_orm(cls, tool: ToolData, update: Optional[dict[str, Any]] = None):
        kwargs = tool.model_dump()

        kwargs['input_handles'] = [
            Handle(name=param.name, description=param.description)
            for param in tool.dynamic_params
            if param.param_type != ParamType.OUTPUT
        ]

        kwargs['output_handles'] = [
            Handle(name=file.name, description=file.description)
            for file in tool.output_files
            if not (file.is_log or file.is_report)
        ]

        if update:
            kwargs.update(update)
        return cls.model_validate(kwargs)


class ToolTaskInfo(BaseModel):
    uid: UUID7
    name: str

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    mkdir_output: bool
    use_temp_dir: bool

    docker_config: Optional[DockerConfig]
