from typing import Any, Optional

from pydantic import BaseModel, UUID7, Field
from uuid_extensions import uuid7

from app.models.tools import ToolTag, ToolData, ToolType, ParamType, ParamDefine, OutputFile, ImageConfig, DockerConfig


class AIGenTool(BaseModel):
    """工具封装定义"""

    name: str = Field(description='工具名称')
    description: str = Field(description='工具功能的简要描述')

    command_template: str = Field(
        description='完整的工具指令模板字符串，可以包含dynamic_params、static_params、position_params、log这几个变量'
    )
    dynamic_params: list[ParamDefine] = Field(description='需要涉及到文件名等要素的变量，包括参数指定的和位置指定的')
    static_params: str = Field(description='与外部文件无关的变量')
    output_files: list[OutputFile] = Field('软件的期望输出文件')

    mkdir_output: bool = Field(description='是否创建输出目录', default=True)
    use_temp_dir: bool = Field(description='是否使用临时目录', default=False)


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
                image=ImageConfig(repository=self.repository, tag=self.tag), docker_settings=self.docker_config
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

    input_handles: list[Handle] = Field(default_factory=list)
    output_handles: list[Handle] = Field(default_factory=list)

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
