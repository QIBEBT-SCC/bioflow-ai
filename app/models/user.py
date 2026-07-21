from enum import IntEnum
from typing import Optional

from sqlalchemy import Column
from sqlmodel import Field, SQLModel

from app.models.decorator.special_field import IntEnumType


class UserRole(IntEnum):
    VISITOR = 0
    MEMBER = 1
    ADMIN = 2


class User(SQLModel, table=True):
    __tablename__ = 'users'

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(default='')
    email: str = Field(unique=True)
    role: UserRole = Field(default=UserRole.VISITOR, sa_column=Column(IntEnumType(UserRole), nullable=False))
    hashed_password: str
    is_active: bool = Field(default=True)
