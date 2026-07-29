import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import frontmatter
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILLS_DIR = PROJECT_ROOT / 'presets' / 'skills'
SKILL_FILE_NAME = 'SKILL.md'
SKILL_NAME_PATTERN = re.compile(r'^[a-z0-9]+(?:-[a-z0-9]+)*$')
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024

BASE_PROMPT = """

## Skills

The following skills are available for this agent. Skills use progressive disclosure:

1. Use each skill's name and description below to decide whether it applies.
2. When a skill applies, call `load_skill` before acting and follow its instructions.
3. If the loaded instructions refer to bundled files, call `read_skill_resource` only for the files needed.

Available skills:
"""


@dataclass(frozen=True)
class SkillDefinition:
    """A discovered Agent Skill."""

    name: str
    description: str
    directory: Path
    instructions: str


def _load_skill_definition(skill_directory: Path) -> SkillDefinition:
    skill_path = skill_directory / SKILL_FILE_NAME
    if not skill_path.is_file():
        msg = f'Skill directory does not contain {SKILL_FILE_NAME}: {skill_directory}'
        raise FileNotFoundError(msg)

    post = frontmatter.load(skill_path)
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
    )


def _discover_skills(agent_name: str, skills_dir: Path) -> dict[str, SkillDefinition]:
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
    for skill_directory in sorted(path for path in agent_skills_dir.iterdir() if path.is_dir()):
        if not skill_directory.resolve().is_relative_to(resolved_agent_dir):
            msg = f'Skill directory must stay within the agent directory: {skill_directory}'
            raise ValueError(msg)
        skill = _load_skill_definition(skill_directory)
        if skill.name in skills:
            msg = f'Duplicate skill name for agent {agent_name!r}: {skill.name}'
            raise ValueError(msg)
        skills[skill.name] = skill

    return skills


def _resolve_skill_resource(skill: SkillDefinition, resource_path: str) -> Path:
    relative_path = Path(resource_path)
    if relative_path.is_absolute():
        msg = 'Skill resource paths must be relative to the skill directory.'
        raise ValueError(msg)

    resolved_path = (skill.directory / relative_path).resolve()
    if not resolved_path.is_relative_to(skill.directory):
        msg = 'Skill resource paths cannot leave the skill directory.'
        raise ValueError(msg)
    if resolved_path == skill.directory / SKILL_FILE_NAME:
        msg = 'Use load_skill to read SKILL.md.'
        raise ValueError(msg)
    if not resolved_path.is_file():
        msg = f'Skill resource was not found: {resource_path}'
        raise FileNotFoundError(msg)
    return resolved_path


class SkillMiddleware(AgentMiddleware):
    """Discover and progressively load the skills assigned to one agent."""

    def __init__(self, agent_name: str, skills_dir: Path = SKILLS_DIR):
        self.agent_name = agent_name
        self.skills = _discover_skills(agent_name, skills_dir)
        self.tools = self._build_tools()

        skill_lines = [f'- `{skill.name}`: {skill.description}' for skill in self.skills.values()]
        self.prompt = f'{BASE_PROMPT}\n' + '\n'.join(skill_lines) if skill_lines else ''

    def _build_tools(self) -> list[BaseTool]:
        skills = self.skills

        async def load_skill(skill_name: str) -> str:
            """Load the full instructions for an available skill.

            Args:
                skill_name: Skill name exactly as advertised in the system prompt.
            """
            skill = skills.get(skill_name)
            if skill is None:
                available = ', '.join(skills) or 'none'
                return f'Skill {skill_name!r} was not found. Available skills: {available}.'
            return skill.instructions

        async def read_skill_resource(skill_name: str, resource_path: str) -> str:
            """Read one file bundled with an available skill.

            Args:
                skill_name: Skill name exactly as advertised in the system prompt.
                resource_path: Relative path named by SKILL.md, such as references/FORMAT.md.
            """
            skill = skills.get(skill_name)
            if skill is None:
                available = ', '.join(skills) or 'none'
                return f'Skill {skill_name!r} was not found. Available skills: {available}.'
            try:
                path = _resolve_skill_resource(skill, resource_path)
            except (FileNotFoundError, ValueError) as exc:
                return str(exc)
            try:
                return path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                return f'Skill resource is not a UTF-8 text file: {resource_path}'

        return [
            tool('load_skill', parse_docstring=True)(load_skill),
            tool('read_skill_resource', parse_docstring=True)(read_skill_resource),
        ]

    def wrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        content_blocks = request.system_message.content_blocks if request.system_message is not None else []
        new_content = cast(
            'list[dict[Any, Any] | str]',
            [*list(content_blocks), {'type': 'text', 'text': self.prompt}],
        )
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        content_blocks = request.system_message.content_blocks if request.system_message is not None else []
        new_content = cast(
            'list[dict[Any, Any] | str]',
            [*list(content_blocks), {'type': 'text', 'text': self.prompt}],
        )
        new_system_message = SystemMessage(content=new_content)
        return await handler(request.override(system_message=new_system_message))
