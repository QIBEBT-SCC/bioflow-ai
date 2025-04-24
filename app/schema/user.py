from enum import IntEnum
from typing import Optional

from pydantic import BaseModel
from sqlmodel import SQLModel, Field


class UserRole(IntEnum):
    VISITOR = 0
    MEMBER = 1
    ADMIN = 2


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class User(SQLModel):
    username: str = Field(default='')
    email: str = Field(index=True, unique=True)
    role: UserRole = Field(default=UserRole.VISITOR)


class UserUpdate(SQLModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
