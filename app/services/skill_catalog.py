import os
import re
import stat
import tempfile
from pathlib import Path

from app.schemas.skill import AgentSkillsPublic, SkillCatalogPublic, SkillPublic, SkillResourcePublic, SkillSummary
from llm.skill_catalog import (
    SKILL_FILE_NAME,
    SKILLS_DIR,
    SkillDefinition,
    discover_agent_skills,
    load_skill_definition,
    parse_skill_definition,
    resolve_skill_resource,
)

AGENT_NAME_PATTERN = re.compile(r'^[a-z0-9]+(?:[_-][a-z0-9]+)*$')
TEMP_FILE_PREFIX = '.skill-write-'
TEMP_DIRECTORY_PREFIX = '.skill-create-'


class SkillCatalogError(ValueError):
    """Base error for invalid skill catalog operations."""


class SkillNotFoundError(SkillCatalogError):
    """Raised when an agent, skill, or resource does not exist."""


class SkillConflictError(SkillCatalogError):
    """Raised when creating a skill that already exists."""


def _agent_directory(agent_name: str, *, skills_dir: Path, must_exist: bool = True) -> Path:
    if not AGENT_NAME_PATTERN.fullmatch(agent_name):
        msg = f'Invalid agent name: {agent_name}'
        raise SkillCatalogError(msg)

    root = skills_dir.resolve()
    agent_directory = (root / agent_name).resolve()
    if not agent_directory.is_relative_to(root):
        msg = 'Agent directory must stay within the skills directory.'
        raise SkillCatalogError(msg)
    if must_exist and not agent_directory.is_dir():
        msg = f'Agent skill directory was not found: {agent_name}'
        raise SkillNotFoundError(msg)
    return agent_directory


def _resource_paths(skill: SkillDefinition) -> list[str]:
    resources: list[str] = []
    for path in skill.directory.rglob('*'):
        if not path.is_file() or path.name.startswith(TEMP_FILE_PREFIX) or path.name == SKILL_FILE_NAME:
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(skill.directory):
            msg = f'Skill resource must stay within its skill directory: {path}'
            raise SkillCatalogError(msg)
        resources.append(resolved_path.relative_to(skill.directory).as_posix())
    return sorted(resources)


def _skill_public(agent_name: str, skill: SkillDefinition) -> SkillPublic:
    return SkillPublic(
        agent_name=agent_name,
        name=skill.name,
        description=skill.description,
        content=skill.content,
        resources=_resource_paths(skill),
    )


def _get_skill(agent_name: str, skill_name: str, *, skills_dir: Path) -> SkillDefinition:
    _agent_directory(agent_name, skills_dir=skills_dir)
    try:
        skills = discover_agent_skills(agent_name, skills_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SkillCatalogError(str(exc)) from exc
    skill = skills.get(skill_name)
    if skill is None:
        msg = f'Skill was not found: {agent_name}/{skill_name}'
        raise SkillNotFoundError(msg)
    return skill


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=TEMP_FILE_PREFIX,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8', newline='') as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(mode)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def list_catalog(*, skills_dir: Path = SKILLS_DIR) -> SkillCatalogPublic:
    """Return all agent directories and their automatically discovered skills."""
    if not skills_dir.is_dir():
        msg = f'Skills directory was not found: {skills_dir}'
        raise SkillNotFoundError(msg)

    agents: list[AgentSkillsPublic] = []
    for agent_directory in sorted(
        path for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith('.')
    ):
        agent_name = agent_directory.name
        _agent_directory(agent_name, skills_dir=skills_dir)
        try:
            skills = discover_agent_skills(agent_name, skills_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SkillCatalogError(str(exc)) from exc
        agents.append(
            AgentSkillsPublic(
                agent_name=agent_name,
                skills=[
                    SkillSummary(
                        name=skill.name,
                        description=skill.description,
                        resource_count=len(_resource_paths(skill)),
                    )
                    for skill in skills.values()
                ],
            )
        )
    return SkillCatalogPublic(agents=agents)


def get_skill(agent_name: str, skill_name: str, *, skills_dir: Path = SKILLS_DIR) -> SkillPublic:
    """Return one complete skill for editing."""
    return _skill_public(agent_name, _get_skill(agent_name, skill_name, skills_dir=skills_dir))


def create_skill(
    agent_name: str,
    skill_name: str,
    content: str,
    *,
    skills_dir: Path = SKILLS_DIR,
) -> SkillPublic:
    """Create and validate a skill under an existing agent directory."""
    agent_directory = _agent_directory(agent_name, skills_dir=skills_dir)
    skill_directory = agent_directory / skill_name
    if skill_directory.exists():
        msg = f'Skill already exists: {agent_name}/{skill_name}'
        raise SkillConflictError(msg)

    try:
        parse_skill_definition(skill_directory, content)
    except ValueError as exc:
        raise SkillCatalogError(str(exc)) from exc

    temporary_directory = Path(tempfile.mkdtemp(dir=agent_directory, prefix=TEMP_DIRECTORY_PREFIX))
    try:
        _atomic_write(temporary_directory / SKILL_FILE_NAME, content)
        temporary_directory.replace(skill_directory)
        skill = load_skill_definition(skill_directory)
    except FileExistsError as exc:
        msg = f'Skill already exists: {agent_name}/{skill_name}'
        raise SkillConflictError(msg) from exc
    except Exception:
        raise
    finally:
        (temporary_directory / SKILL_FILE_NAME).unlink(missing_ok=True)
        if temporary_directory.exists():
            temporary_directory.rmdir()
    return _skill_public(agent_name, skill)


def update_skill(
    agent_name: str,
    skill_name: str,
    content: str,
    *,
    skills_dir: Path = SKILLS_DIR,
) -> SkillPublic:
    """Validate and atomically replace an existing SKILL.md."""
    skill = _get_skill(agent_name, skill_name, skills_dir=skills_dir)
    try:
        parse_skill_definition(skill.directory, content)
    except ValueError as exc:
        raise SkillCatalogError(str(exc)) from exc
    _atomic_write(skill.directory / SKILL_FILE_NAME, content)
    return _skill_public(agent_name, load_skill_definition(skill.directory))


def get_resource(
    agent_name: str,
    skill_name: str,
    resource_path: str,
    *,
    skills_dir: Path = SKILLS_DIR,
) -> SkillResourcePublic:
    """Read one UTF-8 skill resource."""
    skill = _get_skill(agent_name, skill_name, skills_dir=skills_dir)
    try:
        path = resolve_skill_resource(skill.directory, resource_path)
    except FileNotFoundError as exc:
        raise SkillNotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise SkillCatalogError(str(exc)) from exc
    try:
        content = path.read_text(encoding='utf-8')
    except UnicodeDecodeError as exc:
        msg = f'Skill resource is not a UTF-8 text file: {resource_path}'
        raise SkillCatalogError(msg) from exc
    return SkillResourcePublic(path=resource_path, content=content)


def update_resource(
    agent_name: str,
    skill_name: str,
    resource_path: str,
    content: str,
    *,
    skills_dir: Path = SKILLS_DIR,
) -> SkillResourcePublic:
    """Add or atomically replace one UTF-8 text resource."""
    skill = _get_skill(agent_name, skill_name, skills_dir=skills_dir)
    try:
        path = resolve_skill_resource(skill.directory, resource_path, must_exist=False)
    except ValueError as exc:
        raise SkillCatalogError(str(exc)) from exc
    if path.exists() and not path.is_file():
        msg = f'Skill resource path is not a file: {resource_path}'
        raise SkillCatalogError(msg)
    _atomic_write(path, content)
    return SkillResourcePublic(path=resource_path, content=content)
