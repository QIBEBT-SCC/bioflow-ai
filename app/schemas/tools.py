from typing import Optional

from pydantic import BaseModel, UUID7, Field, model_validator
from uuid_extensions import uuid7

from app.models.tools import ToolTag, ToolImage, ToolData, ToolType, ParamType, ParamDefine, OutputFile, ToolHelpDoc
from app.utils.input_format import build_command_template


class AIGenProp(BaseModel):
    name: str
    description: str
    help_command: str

    repository: str
    tag: str = 'latest'


class AIGenTool(BaseModel):
    """工具封装定义"""

    name: str = Field(description='工具名称')
    description: str = Field(description='工具功能的简要描述')

    command_template: str = Field(
        description='完整的工具指令模板字符串，可以包含file_params、static_params、position_params、log这几个变量'
    )
    file_params: list[ParamDefine] = Field(description='需要涉及到文件名等要素的变量，包括参数指定的和位置指定的')
    static_params: str = Field(description='与外部文件无关的变量')
    output_files: list[OutputFile] = Field('软件的期望输出文件')

    mkdir_output: bool = Field(description='是否创建输出目录', default=True)
    use_temp_dir: bool = Field(description='是否使用临时目录', default=False)


class DockerToolCreate(BaseModel):
    """创建Docker工具的请求模型"""

    name: str
    image_id: int
    description: str
    group_id: int
    tags: list[ToolTag]

    help_doc_id: int
    command_template: str
    file_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    mkdir_output: bool = True
    use_temp_dir: bool = False

    def to_tool_data(self) -> ToolData:
        tool_id = uuid7(as_type='str')
        return ToolData(
            uid=str(tool_id),
            name=self.name,
            image_id=self.image_id,
            description=self.description,
            tool_type=ToolType.DOCKER,
            group_id=self.group_id,
            help_doc_id=self.help_doc_id,
            command_template=self.command_template,
            file_params=self.file_params,
            static_params=self.static_params,
            output_files=self.output_files,
            mkdir_output=self.mkdir_output,
            use_temp_dir=self.use_temp_dir,
            tags=self.tags,
        )


class SimpleToolPublic(BaseModel):
    uid: UUID7
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    image: Optional[ToolImage]

    tags: list[ToolTag]


class ToolPublic(BaseModel):
    uid: UUID7
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    command_template: str
    file_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    complete_command: Optional[str] = None

    mkdir_output: bool
    use_temp_dir: bool

    tags: list['ToolTag']
    help_doc: ToolHelpDoc
    image: ToolImage

    @model_validator(mode='after')
    def build_command(self):
        self.complete_command = build_command_template(
            command_template=self.command_template,
            file_params=self.file_params,
            static_params=self.static_params,
            output_files=self.output_files,
        )
        return self


class Handle(BaseModel):
    name: str
    description: str


class ToolArgPublic(BaseModel):
    uid: UUID7
    name: str
    description: str

    file_params: list[ParamDefine] = Field(exclude=True)
    output_files: list[OutputFile] = Field(exclude=True)

    input_handles: list[Handle] = Field(default_factory=list)
    output_handles: list[Handle] = Field(default_factory=list)

    static_params: str

    @model_validator(mode='after')
    def build_handles(self):
        self.input_handles = [
            Handle(name=param.name, description=param.description)
            for param in self.file_params
            if param.param_type != ParamType.OUTPUT
        ]

        self.output_handles = [
            Handle(name=file.name, description=file.description)
            for file in self.output_files
            if not (file.is_log or file.is_report)
        ]
        return self


class ToolTaskInfo(BaseModel):
    uid: UUID7
    name: str

    command_template: str
    file_params: list[ParamDefine]
    static_params: str
    output_files: list[OutputFile]

    mkdir_output: bool
    use_temp_dir: bool

    image: Optional[ToolImage]


class ToolInfo4Task(BaseModel):
    uid: UUID7
    name: str
    description: str

    output_files: list[OutputFile]
