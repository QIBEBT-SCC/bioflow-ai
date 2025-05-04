from typing import Annotated

from fastapi import APIRouter, Depends
from loguru import logger
from pydantic import BaseModel

from app.crud import tag as tag_crud
from app.crud import project as project_crud
from app.core.session import SessionDep
from app.models.project import Project, ProjectPublic
from app.models.user import User
from app.utils.security import get_current_active_user

router = APIRouter()


class ProjectProp(BaseModel):
    name: str
    description: str = ''
    starred: bool = False
    public: bool = True
    tag_ids: list[int] = []


@router.post('')
def new_project(
    project_prop: ProjectProp,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    tags = []
    if project_prop.tag_ids:
        tags = tag_crud.get_all_by_id(session, project_prop.tag_ids)

    project = Project(
        owner_id=current_user.id,
        name=project_prop.name,
        description=project_prop.description,
        starred=project_prop.starred,
        public=project_prop.public,
        tags=tags,
    )
    project_crud.insert(session, project)
    return project


@router.get('', response_model=list[ProjectPublic])
def get_project_list(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
):
    projects = project_crud.get_all(session)
    return [ProjectPublic.from_orm(p) for p in projects]
