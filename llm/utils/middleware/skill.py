from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

import frontmatter
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from llm.utils.formatters import global_mark_formatter

CURRENT_DIR = Path(__file__).parent
SKILLS_DIR = CURRENT_DIR.parent.parent / 'skills'

BASE_PROMPT = """

## Advanced Skills
We provide some advanced skills. When you need them, you can view their detailed descriptions by calling the `view_skill` tool:
"""


@tool(parse_docstring=True)
async def view_skill(skill_name: str) -> str:
    """View the details of a specific skill by its name.

    Args:
        skill_name: The name of the skill to view.
    """
    skill_path = SKILLS_DIR / f'{skill_name}.md'
    if skill_path.exists() and skill_path.is_file():
        skill_data = skill_path.read_text(encoding='utf-8')
        post = frontmatter.loads(skill_data)
        return post.content

    return 'This skill was not found. Please verify the name is correct.'


@tool(parse_docstring=True)
async def assist_tool(
    action: str,
    parameter: Optional[dict[str, Any]],  # noqa: ARG001
    runtime: ToolRuntime,
) -> str:
    """Assist tool for various auxiliary functions.

    This tool is designed for accessibility in SKILL. Do not call it if you do not see information on how to use it.

    Args:
        action: The action to perform.
        parameter: Additional parameter for the action, if needed.
    """
    if action == 'view_global_marks':
        global_marks = runtime.state.get('global_marks', [])
        return global_mark_formatter(global_marks)

    return f'Action "{action}" is not supported.'


class SkillMiddleware(AgentMiddleware):
    tools = [view_skill, assist_tool]  # noqa: RUF012

    def __init__(self, available_skills: list[str]):
        skills: list[str] = []

        # 加载skills
        for file_name in available_skills:
            skill_path = SKILLS_DIR / f'{file_name}.md'
            if skill_path.exists() and skill_path.is_file():
                skill_data = skill_path.read_text(encoding='utf-8')
                post = frontmatter.loads(skill_data)
                skills.append(f'- {post["name"]}: {post["description"]}')
            else:
                msg = f'{skill_path}'
                raise FileNotFoundError(msg)

        self.prompt = BASE_PROMPT + '\n'.join(skills) if skills else ''

    def wrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        new_content = [*list(request.system_message.content_blocks), {'type': 'text', 'text': self.prompt}]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(  # noqa: D102
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        new_content = [*list(request.system_message.content_blocks), {'type': 'text', 'text': self.prompt}]
        new_system_message = SystemMessage(content=new_content)
        return await handler(request.override(system_message=new_system_message))
