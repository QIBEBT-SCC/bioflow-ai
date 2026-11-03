import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, event
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column, relationship

from bioflow_ai.models.base import Base
from bioflow_ai.models.decorator.special_field import IntEnumType, PydanticJson
from bioflow_ai.schemas import tool_config
from bioflow_ai.services.tools.input_format import build_command_template
from bioflow_ai.utils.system import random_uid, utc_now


class ToolGroup(Base):
    __tablename__ = 'tool_groups'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    parent_id: Mapped[int | None] = mapped_column(ForeignKey('tool_groups.id'), default=None)
    tag_id: Mapped[int | None] = mapped_column(ForeignKey('tool_tags.id'), default=None)

    tag: Mapped['ToolTag | None'] = relationship()


class ToolTagLink(Base):
    __tablename__ = 'tool_tag_link'

    tool_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey('tools.uid'), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey('tool_tags.id'), primary_key=True)


class ToolTag(Base):
    __tablename__ = 'tool_tags'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)

    tools: Mapped[list['ToolDefine']] = relationship(
        secondary='tool_tag_link',
        back_populates='tags',
    )


class ToolImage(Base):
    __tablename__ = 'images'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    ref: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    version: Mapped[str]
    description: Mapped[str]
    homepage: Mapped[str | None] = mapped_column(default=None)
    paper_link: Mapped[str | None] = mapped_column(default=None)
    image: Mapped[tool_config.ImageConfig] = mapped_column(PydanticJson(tool_config.ImageConfig), nullable=True)

    tools: Mapped[list['ToolDefine']] = relationship(
        back_populates='image',
        order_by='ToolDefine.name',
    )
    docs: Mapped[list['ToolHelpDoc']] = relationship(back_populates='image')


class ToolHelpDoc(Base):
    __tablename__ = 'tool_docs'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    help_command: Mapped[str]
    description: Mapped[str | None] = mapped_column(default=None)
    content: Mapped[str]

    image_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('images.uid'), default=None)

    image: Mapped['ToolImage | None'] = relationship(back_populates='docs')
    tools: Mapped[list['ToolDefine']] = relationship(back_populates='help_doc')


class ToolDefine(Base):
    __tablename__ = 'tools'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    name: Mapped[str]
    image_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('images.uid'), default=None)
    description: Mapped[str]
    tool_type: Mapped[tool_config.ToolType] = mapped_column(IntEnumType(tool_config.ToolType), nullable=False)
    group_id: Mapped[int | None] = mapped_column(ForeignKey('tool_groups.id'), default=None)
    tags: Mapped[list['ToolTag']] = relationship(
        secondary='tool_tag_link',
        back_populates='tools',
    )

    help_doc_uid: Mapped[uuid.UUID | None] = mapped_column(ForeignKey('tool_docs.uid'), default=None)
    command_template: Mapped[str]
    dynamic_params: Mapped[list[tool_config.ParamDefine]] = mapped_column(
        PydanticJson(list[tool_config.ParamDefine]),
        nullable=True,
    )
    immutable_static_params: Mapped[str | None] = mapped_column(default=None)
    modifiable_static_params: Mapped[str | None] = mapped_column(default=None)
    file_mounts: Mapped[list[tool_config.FileMount]] = mapped_column(
        PydanticJson(list[tool_config.FileMount]),
        nullable=True,
    )
    complete_command: Mapped[str] = mapped_column(default='')

    help_doc: Mapped['ToolHelpDoc | None'] = relationship(back_populates='tools')
    image: Mapped['ToolImage | None'] = relationship(back_populates='tools')


class ToolMemory(Base):
    """A human-approved usage lesson shared by all users of one tool definition."""

    __tablename__ = 'tool_memories'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    tool_uid: Mapped[uuid.UUID] = mapped_column(ForeignKey('tools.uid', ondelete='CASCADE'), index=True)
    description: Mapped[str] = mapped_column(String(120))
    content: Mapped[str] = mapped_column(Text, deferred=True)
    created_by: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


@event.listens_for(ToolDefine, 'before_insert')
@event.listens_for(ToolDefine, 'before_update')
def _render_complete_command(_mapper: Mapper[ToolDefine], _connection: Connection, tool: ToolDefine) -> None:
    """Keep the searchable rendered command aligned with the command definition."""
    tool.complete_command = build_command_template(
        command_template=tool.command_template,
        dynamic_params=tool.dynamic_params or [],
        immutable_static_params=tool.immutable_static_params,
        modifiable_static_params=tool.modifiable_static_params,
        file_mounts=tool.file_mounts or [],
    )
