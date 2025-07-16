from enum import IntEnum
from typing import Optional

from pydantic import UUID7, BaseModel, Field as PField, ValidationError
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
    """Command line argument definition"""

    name: str = PField(
        description='Parameter name used as a formatting keyword in template strings, only lowercase letters and underscores are allowed'
    )
    command: str = PField(
        description='Specific parameter definition, template string constructed using the value of the name field. For example, "-i {name}"'
    )
    param_type: int = PField(
        description='Parameter type, 0 represents INPUT, 1 represents INPUT_POSITION, 2 represents OUTPUT'
    )
    is_file: bool = PField(description='Does the parameter involve a file or a folder?')
    index: Optional[int] = PField(
        description='Position order of positional parameters (only takes effect for parameters of INPUT_POSITION type)',
        default=None,
    )
    required: bool = PField(description='Is it a mandatory parameter for the program to run?', default=True)
    mount_path: Optional[str] = PField(description='Mapped directories in container', default=None)
    description: Optional[str] = PField(description='Parameter Description', default=None)


class OutputFile(BaseModel):
    """Output file definition"""

    name: str = PField(description='Only lowercase letters and underscores are allowed')
    file_path: str = PField(
        description='File path template, only allowed to contain output_dir and sample_name these two variable names'
    )
    is_report: bool = PField(description='Is it as a report file?', default=False)
    is_log: bool = PField(description='Is it a log file?', default=False)
    mount_path: Optional[str] = PField(description='File mapping path in container', default=None)
    description: Optional[str] = PField(description='File description', default=None)


class ImageConfig(BaseModel):
    registry: str = 'docker.io'
    namespace: str = 'library'
    repository: str
    tag: str = 'latest'

    def get_image(self) -> str:
        return f'{self.registry}/{self.namespace}/{self.repository}:{self.tag}'

    @staticmethod
    def from_str(image_str: str):
        try:
            image_group = image_str.split(':')
            assert len(image_group) <= 2

            if len(image_group) == 1:
                tag = 'latest'
            else:
                tag = image_group[1]

            url_group = image_group[0].split('/')
            match len(url_group):
                case 3:
                    return ImageConfig(registry=url_group[0], namespace=url_group[1], repository=url_group[2], tag=tag)
                case 2:
                    return ImageConfig(namespace=url_group[0], repository=url_group[1], tag=tag)
                case 1:
                    return ImageConfig(repository=url_group[0], tag=tag)
                case _:
                    raise AssertionError
        except AssertionError:
            raise ValidationError('Incorrect image format. It MUST be like `[registry]/[namespace]/<repository>:[tag]`')


class ToolGroup(SQLModel, table=True):
    __tablename__ = 'tool_groups'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')
    tag_id: Optional[int] = Field(default=None, foreign_key='tool_tags.id')

    tag: Optional['ToolTag'] = Relationship()


class ToolTagLink(SQLModel, table=True):
    __tablename__ = 'tool_tag_link'

    tool_id: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), ForeignKey('tools.uid'), primary_key=True))
    tag_id: int = Field(foreign_key='tool_tags.id', primary_key=True)


class ToolTag(SQLModel, table=True):
    __tablename__ = 'tool_tags'
    model_config = {'ignored_types': (hybrid_property,)}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)

    tools: list['ToolData'] = Relationship(back_populates='tags', link_model=ToolTagLink)


class ToolImage(SQLModel, table=True):
    __tablename__ = 'images'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    version: str
    description: str
    homepage: str
    paper_link: str
    image: Optional[ImageConfig] = Field(sa_column=Column(PydanticJson(ImageConfig)))

    tools: list['ToolData'] = Relationship(back_populates='image')
    # docs: list['ToolHelpDoc'] = Relationship(back_populates='image')


class ToolHelpDoc(SQLModel, table=True):
    __tablename__ = 'tool_doc'

    id: Optional[int] = Field(default=None, primary_key=True)
    help_command: str
    description: Optional[str]
    content: str

    # image_id: Optional[int] = Field(default=None, foreign_key='image.id')
    parent_id: Optional[int] = Field(default=None, foreign_key='tool_doc.id')

    # image: Optional['ToolImage'] = Relationship(back_populates='docs')
    tools: list['ToolData'] = Relationship(back_populates='help_doc')


class ToolData(SQLModel, table=True):
    __tablename__ = 'tools'

    uid: UUID7 = Field(sa_column=Column(UUID(as_uuid=True), primary_key=True))
    name: str
    image_id: Optional[int] = Field(default=None, foreign_key='images.id')
    description: str
    tool_type: ToolType
    group_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')
    tags: list['ToolTag'] = Relationship(back_populates='tools', link_model=ToolTagLink)

    help_doc_id: Optional[int] = Field(default=None, foreign_key='tool_doc.id')
    command_template: str
    file_params: list[ParamDefine] = Field(sa_column=Column(PydanticJson(list[ParamDefine])))
    static_params: str
    output_files: list[OutputFile] = Field(sa_column=Column(PydanticJson(list[OutputFile])))

    mkdir_output: bool
    use_temp_dir: bool

    help_doc: Optional['ToolHelpDoc'] = Relationship(back_populates='tools')
    image: Optional['ToolImage'] = Relationship(back_populates='tools')
