import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.tools import FileMount, ImageConfig, ParamDefine, ToolDefine, ToolHelpDoc, ToolImage, ToolTag, ToolType
from app.utils.input_format import build_command_template


class ToolGroupPublic(BaseModel):
    id: int
    name: str
    parent_id: int | None
    tag_id: int | None
    tool_count: int


class ImagePublic(BaseModel):
    uid: uuid.UUID
    name: str
    version: str | None
    description: str | None
    homepage: str | None
    paper_link: str | None
    image: ImageConfig

    tools: list['SimpleToolPublic']


class ImageUpdate(BaseModel):
    name: str | None = None
    version: str | None = None
    description: str | None = None
    homepage: str | None = None
    paper_link: str | None = None
    image: ImageConfig | None = None


class DocUpdate(BaseModel):
    help_command: str | None = None
    content: str | None = None


class SimpleToolDoc(BaseModel):
    uid: uuid.UUID
    help_command: str


class RunRequest(BaseModel):
    command: str


class DockerToolCreate(BaseModel):
    name: str
    image_uid: uuid.UUID
    description: str
    help_command: str

    group_id: int
    tag_ids: list[int]

    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None
    modifiable_static_params: str | None
    file_mounts: list[FileMount]

    def to_tool_data(self) -> ToolDefine:
        return ToolDefine(
            name=self.name,
            image_uid=self.image_uid,
            description=self.description,
            tool_type=ToolType.DOCKER,
            group_id=self.group_id,
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            immutable_static_params=self.immutable_static_params,
            modifiable_static_params=self.modifiable_static_params,
            file_mounts=self.file_mounts,
        )


class DockerToolUpdate(BaseModel):
    name: str | None = None
    image_uid: uuid.UUID | None = None
    description: str | None = None
    help_doc_uid: uuid.UUID | None = None

    group_id: int | None = None
    tags: list[ToolTag] | None = None

    command_template: str | None = None
    dynamic_params: list[ParamDefine] | None = None
    immutable_static_params: str | None = None
    modifiable_static_params: str | None = None
    file_mounts: list[FileMount] | None = None


class SimpleToolPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    uid: uuid.UUID
    name: str
    description: str
    tool_type: ToolType
    group_id: int | None

    image: ToolImage | None

    tags: list[ToolTag]


class ToolPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    description: str
    tool_type: ToolType
    group_id: int | None

    file_mounts: list[FileMount]

    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None
    modifiable_static_params: str | None
    command_template: str

    complete_command: str | None = None

    tags: list['ToolTag']
    help_doc: ToolHelpDoc
    image: ToolImage

    @model_validator(mode='after')
    def build_command(self):
        self.complete_command = build_command_template(
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            immutable_static_params=self.immutable_static_params,
            modifiable_static_params=self.modifiable_static_params,
            file_mounts=self.file_mounts,
        )
        return self


class Handle(BaseModel):
    name: str
    description: str


class ToolNodePublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    description: str
    tags: list[ToolTag]

    command_template: str = Field(exclude=True)
    dynamic_params: list[ParamDefine] = Field(exclude=True)
    file_mounts: list[FileMount] = Field(exclude=True)

    input_handles: list[Handle] = Field(default_factory=list)
    output_handles: list[Handle] = Field(default_factory=list)

    immutable_static_params: str | None
    modifiable_static_params: str | None
    complete_command: str | None = None

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
        if not any(handle.name == 'workdir' for handle in self.output_handles):
            self.output_handles.append(Handle(name='workdir', description='The working directory of the tool run.'))

        self.complete_command = build_command_template(
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            immutable_static_params=self.immutable_static_params,
            modifiable_static_params=self.modifiable_static_params,
            file_mounts=self.file_mounts,
        )
        return self


class ToolTaskInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str

    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None
    modifiable_static_params: str | None
    file_mounts: list[FileMount]

    image: ToolImage
