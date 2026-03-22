import uuid
from enum import IntEnum
from typing import Literal, Optional, Self

import uuid_extensions
from pydantic import BaseModel, ValidationError
from pydantic import Field as PField
from sqlalchemy import Column
from sqlalchemy.ext.hybrid import hybrid_property
from sqlmodel import Field, Relationship, SQLModel

from app.models.decorator.special_field import PydanticJson


class ToolType(IntEnum):
    DOCKER = 0
    COMMAND_LINE = 1


class ParamDefine(BaseModel):
    description: str
    command: str
    is_position: bool = PField(default=False)
    index: Optional[int] = PField(default=None)


class FileMount(BaseModel):
    name: str
    description: str
    file_type: Literal['INPUT', 'OUTPUT']
    file_path: str
    mount_path: str
    is_report: bool = PField(default=False)
    is_log: bool = PField(default=False)


class ImageConfig(BaseModel):
    registry: str = 'docker.io'
    namespace: str = 'library'
    repository: str
    tag: str = 'latest'

    def get_image(self) -> str:
        """Return docker image url."""
        return f'{self.registry}/{self.namespace}/{self.repository}:{self.tag}'

    @classmethod
    def from_str(cls, image_str: str) -> Self:
        """Obtain ImageConfig object from the image url."""
        image_group = image_str.split(':')
        if len(image_group) > 2:  # noqa: PLR2004
            err_msg = ''
            raise ValidationError(err_msg)

        tag = 'latest' if len(image_group) == 1 else image_group[1]

        url_group = image_group[0].split('/')
        match len(url_group):
            case 3:
                return cls(registry=url_group[0], namespace=url_group[1], repository=url_group[2], tag=tag)
            case 2:
                return cls(namespace=url_group[0], repository=url_group[1], tag=tag)
            case 1:
                return cls(repository=url_group[0], tag=tag)
            case _:
                err_msg = 'Incorrect image format. It MUST be like `[registry]/[namespace]/<repository>:[tag]`'
                raise ValidationError(err_msg)


class ToolGroup(SQLModel, table=True):
    __tablename__ = 'tool_groups'

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')
    tag_id: Optional[int] = Field(default=None, foreign_key='tool_tags.id')

    tag: Optional['ToolTag'] = Relationship()


class ToolTagLink(SQLModel, table=True):
    __tablename__ = 'tool_tag_link'

    tool_uid: uuid.UUID = Field(foreign_key='tools.uid', primary_key=True)
    tag_id: int = Field(foreign_key='tool_tags.id', primary_key=True)


class ToolTag(SQLModel, table=True):
    __tablename__ = 'tool_tags'
    model_config = {'ignored_types': (hybrid_property,)}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)

    tools: list['ToolDefine'] = Relationship(back_populates='tags', link_model=ToolTagLink)


class ToolImage(SQLModel, table=True):
    __tablename__ = 'images'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    version: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    homepage: Optional[str] = Field(default=None)
    paper_link: Optional[str] = Field(default=None)
    image: ImageConfig = Field(sa_column=Column(PydanticJson(ImageConfig)))

    tools: list['ToolDefine'] = Relationship(
        back_populates='image',
        sa_relationship_kwargs={'order_by': 'ToolDefine.name'},
    )
    docs: list['ToolHelpDoc'] = Relationship(back_populates='image')


class ToolHelpDoc(SQLModel, table=True):
    __tablename__ = 'tool_docs'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    help_command: str
    description: Optional[str] = Field(default=None)
    content: str

    image_uid: Optional[uuid.UUID] = Field(default=None, foreign_key='images.uid')

    image: Optional['ToolImage'] = Relationship(back_populates='docs')
    tools: list['ToolDefine'] = Relationship(back_populates='help_doc')


class ToolDefine(SQLModel, table=True):
    __tablename__ = 'tools'

    uid: uuid.UUID = Field(default_factory=uuid_extensions.uuid7, primary_key=True)
    name: str
    image_uid: Optional[uuid.UUID] = Field(default=None, foreign_key='images.uid')
    description: str
    tool_type: ToolType
    group_id: Optional[int] = Field(default=None, foreign_key='tool_groups.id')
    tags: list['ToolTag'] = Relationship(back_populates='tools', link_model=ToolTagLink)

    help_doc_uid: Optional[uuid.UUID] = Field(default=None, foreign_key='tool_docs.uid')
    command_template: str
    dynamic_params: list[ParamDefine] = Field(sa_column=Column(PydanticJson(list[ParamDefine])))
    immutable_static_params: Optional[str] = Field(default=None)
    modifiable_static_params: Optional[str] = Field(default=None)
    file_mounts: list[FileMount] = Field(sa_column=Column(PydanticJson(list[FileMount])))

    help_doc: Optional['ToolHelpDoc'] = Relationship(back_populates='tools')
    image: Optional['ToolImage'] = Relationship(back_populates='tools')
