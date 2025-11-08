from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from app.core.session import SessionDep
from app.crud import project as project_crud
from app.crud import projetc_tag as tag_crud
from app.models.project import (
    Project,
    ProjectTag,
)
from app.models.user import User
from app.schemas.project import ProjectCreateProp, ProjectPublic, TagProp, TagWithCount
from app.utils.security import get_current_active_user

router = APIRouter()
tag_router = APIRouter()


@tag_router.post('')
def new_tag(
    tag_prop: TagProp,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    tag = ProjectTag(name=tag_prop.name, color=tag_prop.color)
    tag_crud.insert(session, tag)

    return tag


@tag_router.get('', response_model=list[TagWithCount])
def get_tag_list(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    tags = tag_crud.get_all(session)
    return tags


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


@router.get('', response_model=list[ProjectPublic])
def get_project_list(
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
    filter_type: Annotated[Optional[str], Query(description='过滤条件: starred, my, all', alias='filter')] = 'all',
    offset: int = 0,
    limit: Annotated[int, Query(le=20)] = 20,
):
    """获取项目列表

    - filter=all: 获取所有可见项目（公开的或用户拥有的）
    - filter=starred: 获取用户收藏的项目
    - filter=my: 获取用户拥有的项目
    """
    match filter_type:
        case 'starred':
            return project_crud.get_all_star(session, limit, offset, current_user.id)
        case 'my':
            return project_crud.get_all_my(session, limit, offset, current_user.id)
        case 'all':
            return project_crud.get_all(session, limit, offset, current_user.id)
        case _:
            return []


@router.get('/recent', response_model=ProjectPublic)
def get_recent_project(session: SessionDep, current_user: Annotated[User, Depends(get_current_active_user)]):
    """获取用户最近更新的项目"""
    return project_crud.get_recent(session, current_user.id)


@router.get('/{project_id}', response_model=ProjectPublic)
def get_project(
    project_id: int,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    return project_crud.get(session, current_user.id, project_id)
