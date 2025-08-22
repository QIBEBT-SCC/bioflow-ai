from typing import Optional

from pydantic import BaseModel, Field, model_validator
from uuid_extensions import uuid7

from app.models.tools import ToolTag, ToolImage, ToolDefine, ToolType, ParamDefine, FileMount, ToolHelpDoc
from app.utils.input_format import build_command_template


class AIGenProp(BaseModel):
    name: str
    description: str
    help_command: str

    repository: str
    tag: str = 'latest'


class DockerToolCreate(BaseModel):
    """创建Docker工具的请求模型"""

    name: str
    image_id: int
    description: str
    group_id: int
    tags: list[ToolTag]

    help_doc_id: int
    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    file_mounts: list[FileMount]

    mkdir_output: bool = True
    use_temp_dir: bool = False

    def to_tool_data(self) -> ToolDefine:
        tool_id = uuid7(as_type='str')
        return ToolDefine(
            uid=str(tool_id),
            name=self.name,
            image_id=self.image_id,
            description=self.description,
            tool_type=ToolType.DOCKER,
            group_id=self.group_id,
            help_doc_id=self.help_doc_id,
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            static_params=self.static_params,
            file_mounts=self.file_mounts,
            mkdir_output=self.mkdir_output,
            use_temp_dir=self.use_temp_dir,
            tags=self.tags,
        )


class SimpleToolPublic(BaseModel):
    id: int
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    image: Optional[ToolImage]

    tags: list[ToolTag]


class ToolPublic(BaseModel):
    id: int
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    file_mounts: list[FileMount]

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
            dynamic_params=self.dynamic_params,
            static_params=self.static_params,
            file_mounts=self.file_mounts,
        )
        return self


class Handle(BaseModel):
    name: str
    description: str


class ToolNodePublic(BaseModel):
    id: int
    name: str
    description: str

    command_template: str = Field(exclude=True)
    dynamic_params: list[ParamDefine] = Field(exclude=True)
    file_mounts: list[FileMount] = Field(exclude=True)

    input_handles: list[Handle] = Field(default_factory=list)
    output_handles: list[Handle] = Field(default_factory=list)

    static_params: str
    complete_command: Optional[str] = None

    @model_validator(mode='after')
    def build_handles(self):
        self.input_handles = [
            Handle(name=file.name, description=file.description)
            for file in self.file_mounts
            if file.file_type == 'INPUT'
        ]

        self.output_handles = [
            Handle(name=file.name, description=file.description)
            for file in self.file_mounts
            if file.file_type == 'OUTPUT' and not (file.is_log or file.is_report)
        ]

        self.complete_command = build_command_template(
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            static_params=self.static_params,
            file_mounts=self.file_mounts,
        )
        return self


class ToolTaskInfo(BaseModel):
    id: int
    name: str

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    file_mounts: list[FileMount]

    mkdir_output: bool
    use_temp_dir: bool

    image: Optional[ToolImage]


class ToolInfo4Task(BaseModel):
    id: int
    name: str
    description: str

    file_mounts: list[FileMount]
