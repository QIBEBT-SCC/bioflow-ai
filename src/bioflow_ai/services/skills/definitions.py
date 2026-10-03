import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from bioflow_ai.core.paths import SKILLS_DIR

SKILL_FILE_NAME = 'SKILL.md'
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class SkillDefinition:
    """A validated Agent Skill loaded from disk."""

    name: str
    description: str
    directory: Path
    instructions: str
    content: str


def parse_skill_definition(skill_directory: Path, content: str) -> SkillDefinition:
    """Validate raw SKILL.md content against its directory name."""
    skill_path = skill_directory / SKILL_FILE_NAME
    try:
        post = frontmatter.loads(content)
    except Exception as exc:
        msg = f'Invalid YAML frontmatter in {skill_path}: {exc}'
        raise ValueError(msg) from exc

    name = post.get('name')
    description = post.get('description')
    if (
        not isinstance(name, str)
        or not SKILL_NAME_PATTERN.fullmatch(name)
        or len(name) > MAX_SKILL_NAME_LENGTH
    ):
        msg = f'Invalid skill name in {skill_path}: use 1-64 lowercase letters, numbers, and hyphens'
        raise ValueError(msg)
    if name != skill_directory.name:
        msg = f'Skill name {name!r} must match its directory name {skill_directory.name!r}'
        raise ValueError(msg)
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_SKILL_DESCRIPTION_LENGTH
    ):
        msg = f'Invalid skill description in {skill_path}: use 1-1024 characters'
        raise ValueError(msg)

    return SkillDefinition(
        name=name,
        description=description.strip(),
        directory=skill_directory.resolve(),
        instructions=post.content.strip(),
        content=content,
    )


def load_skill_definition(skill_directory: Path) -> SkillDefinition:
    """Load and validate one skill directory."""
    skill_path = skill_directory / SKILL_FILE_NAME
    if not skill_path.is_file():
        msg = f'Skill directory does not contain {SKILL_FILE_NAME}: {skill_directory}'
        raise FileNotFoundError(msg)
    return parse_skill_definition(skill_directory, skill_path.read_text(encoding='utf-8'))


def discover_agent_skills(agent_name: str, skills_dir: Path = SKILLS_DIR) -> dict[str, SkillDefinition]:
    """Discover every immediate skill directory belonging to an agent."""
    agent_skills_dir = skills_dir / agent_name
    if not agent_skills_dir.is_dir():
        msg = f'Agent skill directory was not found: {agent_skills_dir}'
        raise FileNotFoundError(msg)

    resolved_skills_dir = skills_dir.resolve()
    resolved_agent_dir = agent_skills_dir.resolve()
    if not resolved_agent_dir.is_relative_to(resolved_skills_dir):
        msg = f'Agent skill directory must stay within the skills directory: {agent_skills_dir}'
        raise ValueError(msg)

    skills: dict[str, SkillDefinition] = {}
    for skill_directory in sorted(
        path for path in agent_skills_dir.iterdir() if path.is_dir() and not path.name.startswith('.')
    ):
        if not skill_directory.resolve().is_relative_to(resolved_agent_dir):
            msg = f'Skill directory must stay within the agent directory: {skill_directory}'
            raise ValueError(msg)
        skill = load_skill_definition(skill_directory)
        if skill.name in skills:
            msg = f'Duplicate skill name for agent {agent_name!r}: {skill.name}'
            raise ValueError(msg)
        skills[skill.name] = skill
    return skills


def resolve_skill_resource(skill_directory: Path, resource_path: str, *, must_exist: bool = True) -> Path:
    """Resolve a resource path while preventing access outside its skill."""
    relative_path = Path(resource_path)
    if relative_path.is_absolute():
        msg = 'Skill resource paths must be relative to the skill directory.'
        raise ValueError(msg)

    resolved_skill_directory = skill_directory.resolve()
    resolved_path = (resolved_skill_directory / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_skill_directory):
        msg = 'Skill resource paths cannot leave the skill directory.'
        raise ValueError(msg)
    if resolved_path == resolved_skill_directory / SKILL_FILE_NAME:
        msg = 'Use load_skill to read SKILL.md.'
        raise ValueError(msg)
    if resolved_path == resolved_skill_directory:
        msg = 'Skill resource path must identify a file.'
        raise ValueError(msg)
    if must_exist and not resolved_path.is_file():
        msg = f'Skill resource was not found: {resource_path}'
        raise FileNotFoundError(msg)
    return resolved_path
