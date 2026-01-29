from pathlib import Path
from typing import Awaitable, Callable

import frontmatter
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.tools import tool

CURRENT_DIR = Path(__file__).parent
SKILLS_DIR = CURRENT_DIR.parent.parent / 'skills'

BASE_PROMPT = """## Advanced Skills
We provide some advanced skills. When you need them, you can view their detailed descriptions by calling the `view_skill` tool:
"""


@tool(parse_docstring=True)
async def view_skill(skill_name: str):
    """View the details of a specific skill by its name.

    Args:
        skill_name: The name of the skill to view.
    """
    skill_path = SKILLS_DIR / f'{skill_name}.md'
    if skill_path.exists() and skill_path.is_file():
        skill_data = skill_path.read_text(encoding='utf-8')
        post = frontmatter.load(skill_data)
        content = post.content
        return content

    return 'This skill was not found. Please verify the name is correct.'


class SkillMiddleware(AgentMiddleware):
    tools = [view_skill]

    def __init__(self, available_skills: list[str]):
        skills: list[str] = []

        # 加载skills
        for file_name in available_skills:
            skill_path = SKILLS_DIR / f'{file_name}.md'
            if skill_path.exists() and skill_path.is_file():
                skill_data = skill_path.read_text(encoding='utf-8')
                post = frontmatter.load(skill_data)
                skills.append(f'- {post["name"]}: {post["description"]}')
            else:
                raise FileNotFoundError(f'{skill_path}')

        self.prompt = BASE_PROMPT + '\n'.join(skills) if skills else ''

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request.system_prompt = request.system_prompt + '\n\n' + self.prompt
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request.system_prompt = request.system_prompt + '\n\n' + self.prompt
        return handler(request)
