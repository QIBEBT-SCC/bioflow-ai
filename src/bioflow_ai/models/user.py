from sqlalchemy.orm import Mapped, mapped_column

from bioflow_ai.models.base import Base
from bioflow_ai.models.decorator.special_field import IntEnumType
from bioflow_ai.schemas.user_config import UserRole as _UserRole


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(default='')
    email: Mapped[str] = mapped_column(unique=True)
    role: Mapped[_UserRole] = mapped_column(
        IntEnumType(_UserRole),
        default=_UserRole.VISITOR,
        nullable=False,
    )
    hashed_password: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
