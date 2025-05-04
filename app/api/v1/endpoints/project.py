from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.crud import tag as tag_crud
from app.crud import project as project_crud
from app.core.session import SessionDep
from app.models.project import Project, ProjectPublic, ProjectUpdate
from app.models.user import User
from app.utils.security import get_current_active_user

router = APIRouter()


class ProjectCreateProp(BaseModel):
    name: str
    description: str = ''
    starred: bool = False
    public: bool = True
    tag_ids: list[int] = []


@router.post('')
def new_project(
    project_prop: ProjectCreateProp,
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


@router.patch('/{project_id}/starred', response_model=ProjectPublic)
def change_project_star(
    project_id: int,
    project_prop: ProjectUpdate,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    project_update = ProjectUpdate(starred=project_prop.starred)
    updated_project = project_crud.update(session, project_id, project_update)
    return ProjectPublic.from_orm(updated_project)


@router.get('', response_model=list[ProjectPublic])
def get_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    projects = project_crud.get_all(session, limit, offset, current_user.id)
    return [ProjectPublic.from_orm(p) for p in projects]

@router.get('/starred', response_model=list[ProjectPublic])
def get_starred_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    projects = project_crud.get_all_star(session, limit, offset, current_user.id)
    return [ProjectPublic.from_orm(p) for p in projects]

@router.get('/my', response_model=list[ProjectPublic])
def get_my_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    projects = project_crud.get_all_my(session, limit, offset, current_user.id)
    return [ProjectPublic.from_orm(p) for p in projects]