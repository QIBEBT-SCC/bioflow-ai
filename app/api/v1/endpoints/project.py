from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.crud import tag as tag_crud
from app.crud import project as project_crud
from app.core.session import SessionDep
from app.models.project import Project, ProjectPublic
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


@router.post('/{project_id}/star')
def star_project(
    project_id: int,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    project_crud.star_project(session, current_user.id, project_id)


@router.post('/{project_id}/unstar')
def unstar_project(
    project_id: int,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    project_crud.unstar_project(session, current_user.id, project_id)


@router.get('')
def get_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[ProjectPublic]:
    return project_crud.get_all(session, limit, offset, current_user.id)


@router.get('/starred')
def get_starred_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
) -> list[ProjectPublic]:
    return project_crud.get_all_star(session, limit, offset, current_user.id)


@router.get('/my', response_model=list[ProjectPublic])
def get_my_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    return project_crud.get_all_my(session, limit, offset, current_user.id)


@router.get('/recent')
def get_recent_project(
    session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]
) -> Optional[ProjectPublic]:
    return project_crud.get_recent(session, current_user.id)


@router.get('/{project_id}')
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Optional[ProjectPublic]:
    return project_crud.get(session, current_user.id, project_id)
