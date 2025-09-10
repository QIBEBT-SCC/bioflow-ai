from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ArgsSchema
from langgraph.types import interrupt
from pydantic import BaseModel, Field


class AskHumanSchema(BaseModel):
    question: str = Field(description='Issues to consult with humans')


class AskHuman(BaseTool):
    name: str = 'ask_human'
    description: str = 'Consult humans for advice'
    args_schema: Optional[ArgsSchema] = AskHumanSchema
    return_direct: bool = False

    def _run(self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        answer = interrupt({'question': question})
        return f'Human feedback is: "{answer}"'


ask_human = AskHuman()
