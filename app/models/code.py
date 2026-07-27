import uuid

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.models.decorator.special_field import PydanticJson
from app.utils.system import random_uid


class CodeDefine(Base):
    """A reusable workflow CODE node definition."""

    __tablename__ = 'codes'

    uid: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=random_uid)
    name: Mapped[str] = mapped_column(index=True)
    description: Mapped[str] = mapped_column(Text)
    node_type: Mapped[str] = mapped_column(index=True)
    code: Mapped[str] = mapped_column(Text)
    dependencies: Mapped[list[str]] = mapped_column(
        PydanticJson(list[str]),
        default=list,
        nullable=False,
    )
