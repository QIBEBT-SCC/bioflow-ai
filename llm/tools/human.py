from typing import Annotated

from langchain_core.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


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
