from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool

from bioflow_ai.services.skills.definitions import (
    SKILLS_DIR,
    SkillDefinition,
    discover_agent_skills,
    resolve_skill_resource,
)

BASE_PROMPT = """

## Skills

The following skills are available for this agent. Skills use progressive disclosure:

1. Use each skill's name and description below to decide whether it applies.
2. When a skill applies, call `load_skill` before acting and follow its instructions.
3. If the loaded instructions refer to bundled files, call `read_skill_resource` only for the files needed.

Available skills:
"""


class SkillMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Discover and progressively load the skills assigned to one agent."""

    def __init__(self, agent_name: str, skills_dir: Path = SKILLS_DIR):
        self.agent_name = agent_name
        self.skills_dir = skills_dir
        self.skills: dict[str, SkillDefinition] = {}
        self.tools = self._build_tools()
        self._refresh()

    def _refresh(self) -> None:
        self.skills = discover_agent_skills(self.agent_name, self.skills_dir)
        skill_lines = [f'- `{skill.name}`: {skill.description}' for skill in self.skills.values()]
        self.prompt = f'{BASE_PROMPT}\n' + '\n'.join(skill_lines) if skill_lines else ''

    def _build_tools(self) -> list[BaseTool]:
        async def load_skill(skill_name: str) -> str:
            """Load the full instructions for an available skill.

            Args:
                skill_name: Skill name exactly as advertised in the system prompt.
            """
            self._refresh()
            skill = self.skills.get(skill_name)
            if skill is None:
                available = ', '.join(self.skills) or 'none'
                return f'Skill {skill_name!r} was not found. Available skills: {available}.'
            return skill.instructions

        async def read_skill_resource(skill_name: str, resource_path: str) -> str:
            """Read one file bundled with an available skill.

            Args:
                skill_name: Skill name exactly as advertised in the system prompt.
                resource_path: Relative path named by SKILL.md, such as references/FORMAT.md.
            """
            self._refresh()
            skill = self.skills.get(skill_name)
            if skill is None:
                available = ', '.join(self.skills) or 'none'
                return f'Skill {skill_name!r} was not found. Available skills: {available}.'
            try:
                path = resolve_skill_resource(skill.directory, resource_path)
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
        self._refresh()
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
        self._refresh()
        content_blocks = request.system_message.content_blocks if request.system_message is not None else []
        new_content = cast(
            'list[dict[Any, Any] | str]',
            [*list(content_blocks), {'type': 'text', 'text': self.prompt}],
        )
        new_system_message = SystemMessage(content=new_content)
        return await handler(request.override(system_message=new_system_message))
