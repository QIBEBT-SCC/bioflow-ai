import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.decorator.special_field import IntEnumType, PydanticJson
from app.schemas import tool_config
from app.utils.system import random_uid


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

    help_doc: Mapped['ToolHelpDoc | None'] = relationship(back_populates='tools')
    image: Mapped['ToolImage | None'] = relationship(back_populates='tools')
