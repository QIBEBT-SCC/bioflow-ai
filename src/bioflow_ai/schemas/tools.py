import uuid
from datetime import datetime
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bioflow_ai.schemas.tool_config import FileMount, ImageConfig, ParamDefine, ToolType
from bioflow_ai.schemas.workflow_run import RunStatus
from bioflow_ai.services.tools.input_format import build_command_template


class ToolTagPublic(BaseModel):
    """A tool tag returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class ToolGroupCreate(BaseModel):
    """Properties accepted when creating a tool group."""

    name: str
    parent_id: int | None = None


class ToolGroupRecord(BaseModel):
    """A persisted tool group."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    parent_id: int | None
    tag_id: int | None


class ToolGroupPublic(BaseModel):
    """A tool group with its tool count."""

    id: int
    name: str
    parent_id: int | None
    tag_id: int | None
    tool_count: int


class ToolImageData(BaseModel):
    """Shared persisted properties of a tool image."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    version: str
    description: str
    homepage: str | None = None
    paper_link: str | None = None
    image: ImageConfig


class ImageCreate(ToolImageData):
    """Properties accepted when creating a tool image."""


class ToolImageImport(ToolImageData):
    """Validated tool image data loaded from a preset."""

    uid: uuid.UUID | None = None
    ref: str


class ToolImagePublic(ToolImageData):
    """A persisted tool image without reverse relationships."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    ref: str


class ImagePublic(ToolImagePublic):
    """A persisted tool image with its tools."""

    tools: list['SimpleToolPublic']


class ImageUpdate(BaseModel):
    """Editable tool image properties."""

    name: str | None = None
    aliases: list[str] = Field(default_factory=list)
    version: str | None = None
    description: str | None = None
    homepage: str | None = None
    paper_link: str | None = None
    image: ImageConfig | None = None


class DocUpdate(BaseModel):
    """Editable tool help document properties."""

    help_command: str | None = None
    content: str | None = None


class ToolHelpDocImport(BaseModel):
    """Validated help document data loaded from a preset."""

    uid: uuid.UUID | None = None
    help_command: str
    description: str | None = None
    content: str
    image_uid: uuid.UUID | None = None


class ToolHelpDocPublic(BaseModel):
    """A persisted tool help document."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    help_command: str
    description: str | None
    content: str
    image_uid: uuid.UUID | None


class SimpleToolDoc(BaseModel):
    """A compact tool help document reference."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    help_command: str


class RunRequest(BaseModel):
    """A command to execute in a tool image."""

    command: str


class DockerToolCreate(BaseModel):
    """Properties accepted when creating a Docker tool."""

    name: str
    tool_type: ToolType = ToolType.DOCKER
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


class ToolDefinitionImport(BaseModel):
    """Validated tool definition data loaded from a preset."""

    uid: uuid.UUID | None = None
    name: str
    image_uid: uuid.UUID | None = None
    description: str
    tool_type: ToolType
    group_id: int | None = None
    help_doc_uid: uuid.UUID | None = None
    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None = None
    modifiable_static_params: str | None = None
    file_mounts: list[FileMount]


class DockerToolUpdate(BaseModel):
    """Editable Docker tool properties."""

    name: str | None = None
    tool_type: ToolType | None = None
    image_uid: uuid.UUID | None = None
    description: str | None = None
    help_doc_uid: uuid.UUID | None = None

    group_id: int | None = None
    tags: list[ToolTagPublic] | None = None

    command_template: str | None = None
    dynamic_params: list[ParamDefine] | None = None
    immutable_static_params: str | None = None
    modifiable_static_params: str | None = None
    file_mounts: list[FileMount] | None = None


class SimpleToolPublic(BaseModel):
    """A compact persisted tool definition."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    name: str
    description: str
    tool_type: ToolType
    group_id: int | None

    image: ToolImagePublic | None

    tags: list[ToolTagPublic]


class ToolPublic(BaseModel):
    """A complete persisted tool definition."""

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

    tags: list[ToolTagPublic]
    help_doc: ToolHelpDocPublic
    image: ToolImagePublic

    @model_validator(mode='after')
    def build_command(self) -> Self:
        """Build the complete command after validation."""
        self.complete_command = build_command_template(
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            immutable_static_params=self.immutable_static_params,
            modifiable_static_params=self.modifiable_static_params,
            file_mounts=self.file_mounts,
        )
        return self


class ToolWorkflowUsage(BaseModel):
    """A saved workflow template that currently references a tool."""

    uid: uuid.UUID
    name: str
    description: str
    public: bool
    node_count: int


class ToolRunUsage(BaseModel):
    """A historical run containing at least one task for a tool."""

    uid: uuid.UUID
    name: str
    project_id: int | None
    project_name: str | None
    workflow_uid: uuid.UUID | None
    workflow_name: str | None
    status: RunStatus
    create_time: datetime | None
    task_count: int


class ToolUsagePublic(BaseModel):
    """Paginated current and historical usage for one tool."""

    workflow_total: int
    run_total: int
    workflow_offset: int
    run_offset: int
    limit: int
    workflows: list[ToolWorkflowUsage]
    runs: list[ToolRunUsage]


class Handle(BaseModel):
    """A visual workflow input or output handle."""

    name: str
    description: str
    shared_writable: bool = False


class ToolNodePublic(BaseModel):
    """A tool definition projected into a workflow node."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    tool_type: ToolType = ToolType.DOCKER
    name: str
    description: str
    tags: list[ToolTagPublic]

    command_template: str = Field(exclude=True)
    dynamic_params: list[ParamDefine] = Field(exclude=True)
    file_mounts: list[FileMount] = Field(exclude=True)

    input_handles: list[Handle] = Field(default_factory=list)
    output_handles: list[Handle] = Field(default_factory=list)

    immutable_static_params: str | None
    modifiable_static_params: str | None
    complete_command: str | None = None

    @model_validator(mode='after')
    def build_handles(self) -> Self:
        """Build visual handles and the complete command after validation."""
        self.input_handles = [
            Handle(name=file.name, description=file.description, shared_writable=file.is_passthrough)
            for file in self.file_mounts
            if file.file_type == 'INPUT'
        ]

        passthrough_inputs_by_mount: dict[str, list[str]] = {}
        if self.tool_type == ToolType.DOCKER_PASSTHROUGH:
            for file in self.file_mounts:
                if file.is_passthrough:
                    mount = PurePosixPath(file.mount_path).as_posix()
                    passthrough_inputs_by_mount.setdefault(mount, []).append(file.name)

        def output_description(file: FileMount) -> str:
            if file.file_type != 'OUTPUT' or self.tool_type != ToolType.DOCKER_PASSTHROUGH:
                return file.description
            mount = PurePosixPath(file.mount_path).as_posix()
            if mount == '/data/output' or mount not in passthrough_inputs_by_mount:
                return file.description
            inputs = ', '.join(passthrough_inputs_by_mount[mount])
            note = f'(Written in the same directory as passthrough input {inputs}, not in workdir.)'
            return f'{file.description} {note}'.strip()

        self.output_handles = [
            Handle(
                name=file.name,
                description=output_description(file),
                shared_writable=self.tool_type == ToolType.DOCKER_PASSTHROUGH,
            )
            for file in self.file_mounts
            if (file.file_type == 'OUTPUT' or file.is_passthrough) and not (file.is_log or file.is_report)
        ]
        if not any(handle.name == 'workdir' for handle in self.output_handles):
            workdir_description = 'The working directory of the tool run.'
            if self.tool_type == ToolType.DOCKER_PASSTHROUGH:
                non_log_outputs = [
                    file
                    for file in self.file_mounts
                    if file.file_type == 'OUTPUT' and not (file.is_log or file.is_report)
                ]
                has_workdir_outputs = any(
                    PurePosixPath(file.mount_path).as_posix() == '/data/output'
                    or PurePosixPath(file.mount_path).as_posix() not in passthrough_inputs_by_mount
                    for file in non_log_outputs
                )
                if non_log_outputs and not has_workdir_outputs:
                    workdir_description = (
                        'The working directory of the tool run (only logs/reports are written here; '
                        'declared outputs are written in their passthrough input directories).'
                    )
            self.output_handles.append(
                Handle(
                    name='workdir',
                    description=workdir_description,
                    shared_writable=self.tool_type == ToolType.DOCKER_PASSTHROUGH,
                )
            )

        self.complete_command = build_command_template(
            command_template=self.command_template,
            dynamic_params=self.dynamic_params,
            immutable_static_params=self.immutable_static_params,
            modifiable_static_params=self.modifiable_static_params,
            file_mounts=self.file_mounts,
        )
        return self


class ToolTaskInfo(BaseModel):
    """Task-relevant metadata for executing a tool."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    tool_type: ToolType = ToolType.DOCKER
    name: str

    command_template: str
    dynamic_params: list[ParamDefine]
    immutable_static_params: str | None
    modifiable_static_params: str | None
    file_mounts: list[FileMount]

    image: ToolImagePublic


class WorkflowToolDefinition(BaseModel):
    """Round-trippable tool snapshot shared by editing and static analysis."""

    model_config = ConfigDict(from_attributes=True)

    uid: uuid.UUID
    tool_type: ToolType = ToolType.DOCKER
    name: str
    description: str
    tags: list[ToolTagPublic]
    command_template: str
    dynamic_params: list[ParamDefine]
    file_mounts: list[FileMount]
    immutable_static_params: str | None
    modifiable_static_params: str | None

    def to_node(self) -> ToolNodePublic:
        """Project the Tool, its execution mode, and automatic file handles."""
        return ToolNodePublic.model_validate(self.model_dump())
