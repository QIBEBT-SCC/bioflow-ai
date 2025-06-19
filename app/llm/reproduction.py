import operator
from operator import itemgetter
from typing import Annotated, Literal

from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.core.llm_chat import gpt_41_mini, gemini_25_flash
from app.llm.prompt.reproduction_prompt import GET_TOOL_USER, CHECK_DB_USER
from app.schemas.tools import SimpleToolPublic

REPRODUCTION_SYS = """你是一名生物信息学分析专家，你的任务是根据原始的文献片段复现出一个可以运行的生物信息数据分析工作流。

<task-description
流程复现是一项较为复杂的工作，因此，你需要将这个工作拆分为多个步骤进行：
1. 首先，提取文章中涉及到的生物信息分析工具名称
</task-description>

<human_breakpoint>
如果任意一步出现了你无法解决的问题，你可以调用`ask_human`工具，他会向人类请求干涉意见
人类的反馈意见具有最高优先级
</human_breakpoint>
"""


class PaperTool(BaseModel):
    name: str = Field(description='Exact name of the tool')
    paper_text: str = Field(description='The original sentence in the paper referring to this tool')


class ToolsOutput(BaseModel):
    """Bioinformatics tools used in the literature"""

    tools: list[PaperTool] = Field(description='All tools used', default_factory=list)
    unsure_tools: list[PaperTool] = Field(description='Tools that you are not sure count', default_factory=list)


def tool_formatter(db_tools: list[SimpleToolPublic]) -> str:
    formatted = [
        (
            '   <tool>\n'
            f'      <db_name>{t.name}</db_name>\n'
            f'      <desc>{t.description}</desc>\n'
            f'      <docker_repo>{t.docker_config.image.repository}</docker_repo>\n'
            f'      <docker_tag>{t.docker_config.image.tag}</docker_tag>\n'
            '   </tool>'
        )
        for t in db_tools
    ]

    return '\n\n'.join(formatted)


async def check_db_tool(paper_db: PaperTool, db_tools: list[SimpleToolPublic]):
    formatter = itemgetter('db_tools') | RunnableLambda(tool_formatter)
    prompt = ChatPromptTemplate.from_messages([('human', CHECK_DB_USER)])
    chain = RunnablePassthrough.assign(tool_info=formatter) | prompt

    print(chain.invoke({'db_tools': db_tools, 'tool_name': paper_db.name, 'paper_text': paper_db.paper_text}))


class AgentState(MessagesState):
    paper_content: str
    price: Annotated[float, operator.add]


class ReproductionAgent:
    def __init__(self):
        self.struct_llm: ChatOpenAI = gpt_41_mini()
        self.tool_llm: ChatOpenAI = gemini_25_flash()

    @staticmethod
    @tool
    def ask_human_message(query: str):
        """Request assistance from a human."""
        human_response = interrupt({'query': query})
        return human_response['data']

    def extract_tools(self, state: AgentState) -> Command[Literal['']]:
        messages = state.get('messages', [])
        if messages:
            prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=REPRODUCTION_SYS), MessagesPlaceholder(variable_name='history')]
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=REPRODUCTION_SYS), ('human', GET_TOOL_USER)]
            )
        chain = prompt | self.struct_llm.with_structured_output(ToolsOutput)

        output: ToolsOutput = chain.invoke({'paper_content': state.get('paper_content')})

        if len(output.tools) == 0:
            return Command(goto='ask_human_message')

        if output.unsure_tools:
            return Command(goto='')

        return Command(
            update={
                'messages': [
                    AIMessage(f'I found {len(output.tools)} tools:[{", ".join([t.name for t in output.tools])}]')
                ]
            }
        )

    def return_node(self,state: AgentState):
        pass

    def build(self, memory: BaseCheckpointSaver) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node('extract_tools', self.extract_tools)
        graph.add_node('extract_tools', self.extract_tools)

        graph.add_edge(START, 'extract_tools')

        return graph.compile(checkpointer=memory)
