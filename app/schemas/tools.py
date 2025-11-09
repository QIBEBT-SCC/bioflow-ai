import uuid
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.models.tools import FileMount, ParamDefine, ToolDefine, ToolHelpDoc, ToolImage, ToolTag, ToolType
from app.utils.input_format import build_command_template


class ToolGroupPublic(BaseModel):
    """工具分组公开信息，包含工具数量"""

    id: int
    name: str
    parent_id: Optional[int]
    tag_id: Optional[int]
    tool_count: int


class ImageUpdate(BaseModel):
    uid: uuid.UUID
    description: Optional[str] = None
    homepage: Optional[str] = None
    paper_link: Optional[str] = None


class SimpleToolDoc(BaseModel):
    uid: uuid.UUID
    help_command: str


class AiGenRequest(BaseModel):
    name: str
    description: str

    image_uid: uuid.UUID


class DockerToolCreate(BaseModel):
    """创建Docker工具的请求模型"""

    name: str
    image_uid: uuid.UUID
    description: str
    help_doc_uid: uuid.UUID

    group_id: int
    tags: list[ToolTag]

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    file_mounts: list[FileMount]
    mkdir_output: bool = True
    use_temp_dir: bool = False

    def to_tool_data(self) -> ToolDefine:
        return ToolDefine(
            name=self.name,
            image_uid=self.image_uid,
            description=self.description,
            tool_type=ToolType.DOCKER,
            group_id=self.group_id,
            help_doc_uid=self.help_doc_uid,
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            static_params=self.static_params,
            file_mounts=self.file_mounts,
            mkdir_output=self.mkdir_output,
            use_temp_dir=self.use_temp_dir,
            tags=self.tags,
        )


class SimpleToolPublic(BaseModel):
    uid: uuid.UUID
    name: str
    description: str
    tool_type: ToolType
    group_id: Optional[int]

    image: Optional[ToolImage]

    tags: list[ToolTag]


class ToolPublic(BaseModel):
    uid: uuid.UUID
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
    uid: uuid.UUID
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
    uid: uuid.UUID
    name: str

    command_template: str
    dynamic_params: list[ParamDefine]
    static_params: str
    file_mounts: list[FileMount]

    mkdir_output: bool
    use_temp_dir: bool

    image: ToolImage
