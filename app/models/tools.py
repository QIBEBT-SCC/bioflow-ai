from enum import IntEnum
from typing import Optional, Any
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import SQLModel, Field


class ToolType(IntEnum):
    DOCKER = 0
    COMMAND_LINE = 1


class ParamType(IntEnum):
    INPUT = 0
    OUTPUT = 1


class ParamDefine(BaseModel):
    """命令行参数定义

    Attributes:
        name: 参数名称
        command: 参数命令格式，如 "-i {value}"
        param_type: 是输入相关参数还是输出相关参数
        is_file: 输入是文件还是文件夹
        required: 是否必选
        default: 默认值
        description: 参数描述
    """

    name: str
    command: str
    param_type: ParamType
    is_file: bool
    required: bool = True
    mount_path: Optional[str] = None
    default: Optional[str] = None
    description: Optional[str] = None


class OutputFile(BaseModel):
    name: str
    file_path: str
    mount_path: Optional[str]


class ImageConfig(SQLModel):
    repository: str
    tag: str = 'latest'

    def get_image(self) -> str:
        return f'{self.repository}:{self.tag}'


class DockerConfig(SQLModel):
    image: ImageConfig
    docker_settings: dict[str, Any]


class ToolData(SQLModel, table=True):
    __tablename__ = 'tools'

    uid: str = Field(primary_key=True)
    name: str
    description: str
    tool_type: ToolType

    command_template: str
    required_params: list[ParamDefine] = Field(default_factory=list, sa_column=Column(JSONB))
    optional_params: str
    output_files: list[OutputFile] = Field(default_factory=list, sa_column=Column(JSONB))

    help_command: str

    mkdir_output: bool
    use_temp_dir: bool

    docker_config: Optional[DockerConfig] = Field(default=None, sa_column=Column(JSONB))


class DockerToolCreate(BaseModel):
    """创建Docker工具的请求模型"""

    name: str
    description: str

    command_template: str
    required_params: list[ParamDefine]
    optional_params: str
    output_files: list[OutputFile]

    help_command: str

    mkdir_output: bool = True
    use_temp_dir: bool = False

    repository: str
    tag: str = 'latest'
    docker_config: dict[str, Any] = {}

    def to_tool_data(self) -> ToolData:
        tool_id = uuid4()
        return ToolData(
            uid=str(tool_id),
            name=self.name,
            description=self.description,
            tool_type=ToolType.DOCKER,
            command_template=self.command_template,
            required_params=self.required_params,
            optional_params=self.optional_params,
            output_files=self.output_files,
            help_command=self.help_command,
            mkdir_output=self.mkdir_output,
            use_temp_dir=self.use_temp_dir,
            docker_config=DockerConfig(
                image=ImageConfig(repository=self.repository, tag=self.tag),
                docker_settings=self.docker_config,
            ),
        )
