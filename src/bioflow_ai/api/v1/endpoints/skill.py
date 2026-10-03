from typing import NoReturn

from fastapi import APIRouter, HTTPException, status

from bioflow_ai.core.auth import AdminDep
from bioflow_ai.schemas.skill import (
    SkillCatalogPublic,
    SkillCreate,
    SkillPublic,
    SkillResourcePublic,
    SkillResourceUpdate,
    SkillUpdate,
)
from bioflow_ai.services import skill_catalog

router = APIRouter()


def _raise_skill_error(error: skill_catalog.SkillCatalogError) -> NoReturn:
    if isinstance(error, skill_catalog.SkillNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(error, skill_catalog.SkillConflictError):
        status_code = status.HTTP_409_CONFLICT
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    raise HTTPException(status_code=status_code, detail=str(error)) from error


@router.get('')
def list_skills(_admin: AdminDep) -> SkillCatalogPublic:
    """List all automatically discovered agent skills."""
    try:
        return skill_catalog.list_catalog()
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)


@router.post('/{agent_name}')
def create_skill(agent_name: str, data: SkillCreate, _admin: AdminDep) -> SkillPublic:
    """Create a skill in an existing agent directory."""
    try:
        return skill_catalog.create_skill(agent_name, data.name, data.content)
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)


@router.get('/{agent_name}/{skill_name}')
def get_skill(agent_name: str, skill_name: str, _admin: AdminDep) -> SkillPublic:
    """Return the complete editable SKILL.md and resource listing."""
    try:
        return skill_catalog.get_skill(agent_name, skill_name)
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)


@router.patch('/{agent_name}/{skill_name}')
def update_skill(agent_name: str, skill_name: str, data: SkillUpdate, _admin: AdminDep) -> SkillPublic:
    """Validate and replace an existing SKILL.md."""
    try:
        return skill_catalog.update_skill(agent_name, skill_name, data.content)
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)


@router.get('/{agent_name}/{skill_name}/resources/{resource_path:path}')
def get_skill_resource(
    agent_name: str,
    skill_name: str,
    resource_path: str,
    _admin: AdminDep,
) -> SkillResourcePublic:
    """Return one editable UTF-8 skill resource."""
    try:
        return skill_catalog.get_resource(agent_name, skill_name, resource_path)
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)


@router.put('/{agent_name}/{skill_name}/resources/{resource_path:path}')
def update_skill_resource(
    agent_name: str,
    skill_name: str,
    resource_path: str,
    data: SkillResourceUpdate,
    _admin: AdminDep,
) -> SkillResourcePublic:
    """Add or replace one UTF-8 skill resource."""
    try:
        return skill_catalog.update_resource(agent_name, skill_name, resource_path, data.content)
    except skill_catalog.SkillCatalogError as error:
        _raise_skill_error(error)
