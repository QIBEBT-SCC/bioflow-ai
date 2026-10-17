from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from bioflow_ai.llm.context import ProjectContext


@tool('ask_human', parse_docstring=True)
def ask_human(question: str) -> str:
    """Consult humans for advice.

    Args:
        question: Issues to consult with humans
    """
    answer = interrupt({'question': question})
    return f'Human feedback is: "{answer}"'


class Question(BaseModel):
    question: str = Field(description='The question to ask the human')
    choices: list[str] = Field(
        min_length=1,
        max_length=4,
        description='One to four concise suggested answers. The user may also provide a custom answer.',
    )


@tool('ask_human', parse_docstring=True)
def ask_human_with_choices(
    questions: Annotated[list[Question], Field(min_length=1, max_length=3)],
) -> str:
    """Ask the user one or more questions with suggested answers.

    Args:
        questions: Questions to ask. Prefer one question at a time and one or two choices per question.
    """
    answer = interrupt(
        {
            'kind': 'questions',
            'questions': [question.model_dump(mode='json') for question in questions],
        }
    )
    return f'Human feedback is:\n{answer}'


@tool
def confirm_plan(runtime: ToolRuntime[ProjectContext]) -> Command:
    """Request human approval of the data analysis plan saved in `PLAN.md`.

    Approval ends the task. Requires an agent state with `should_end` (provided by `EndLoopMiddleware`).
    """
    workdir = Path(runtime.context.workdir)
    missing = [name for name in ('SAMPLES.md', 'PLAN.md') if not (workdir / name).is_file()]
    if missing:
        names = ', '.join(f'`{name}`' for name in missing)
        return Command(
            update={
                'messages': [
                    ToolMessage(
                        content=f'{names} not found in the project directory. Write it before requesting approval.',
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    if not runtime.context.auto_approve:
        content = (workdir / 'PLAN.md').read_text(encoding='utf-8')
        answer = interrupt({'kind': 'plan_approval', 'plan': content})
        if answer is not True:
            feedback = '' if isinstance(answer, bool) else str(answer).strip()
            content = (
                f'The plan was not approved. User feedback is: "{feedback}"'
                if feedback
                else 'The plan was not approved and the user gave no feedback.'
            )
            return Command(
                update={'messages': [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]},
            )

    return Command(
        update={
            'should_end': True,
            'messages': [ToolMessage(content='User confirmed the plan.', tool_call_id=runtime.tool_call_id)],
        },
    )
