from typing import Annotated, Sequence

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.crud import tag as tag_crud
from app.core.session import SessionDep
from app.models.project import Tag
from app.models.user import User
from app.utils.security import get_current_active_user

router = APIRouter()


class TagProp(BaseModel):
    name: str
    color: str


@router.post('')
def new_tag(
    tag_prop: TagProp,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    tag = Tag(name=tag_prop.name, color=tag_prop.color)
    tag_crud.insert(session, tag)

    return tag


@router.get('')
def get_tag_list(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> Sequence[Tag]:
    tags = tag_crud.get_all(session)
    return tags
