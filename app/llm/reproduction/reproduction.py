import operator
from operator import itemgetter
from typing import Annotated, Literal, TypedDict

import rich
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Command, interrupt, Send
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tool as tool_crud
from app.core.llm_chat import gpt_41_mini, gemini_25_flash
from app.core.session import engine
from app.llm.reproduction.reproduction_prompt import CHECK_DB_USER, GET_TOOL_USER, CHECK_DB_SYSTEM, GET_TOOL_SYSTEM
from app.schemas.tools import SimpleToolPublic


class PaperTool(BaseModel):
    name: str = Field(description='Exact name of the tool')
    paper_text: str = Field(description='The original sentence in the paper referring to this tool')


class ToolsOutput(BaseModel):
    """Bioinformatics tools used in the literature"""

    tools: list[PaperTool] = Field(description='All tools used', default_factory=list)
    unsure_tools: list[PaperTool] = Field(description='Tools that you are not sure count', default_factory=list)


class DBChooseResult(BaseModel):
    """"""

    is_right_tool: bool
    selected_tools: list[str]
    missed_tools: list[str]


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


class AgentState(MessagesState):
    # temp_messages: Annotated[list[AnyMessage], add_messages]

    paper_content: str
    paper_tools: ToolsOutput
    check_result: Annotated[list, operator.add]

    price: Annotated[float, operator.add]


class CheckState(TypedDict):
    name: str
    content: str


class ReproductionAgent:
    def __init__(self):
        self.struct_llm: ChatOpenAI = gpt_41_mini()
        self.tool_llm: ChatGoogleGenerativeAI = gemini_25_flash()

    @staticmethod
    @tool
    def ask_human_message(query: str):
        """Request assistance from a human."""
        human_response = interrupt({'query': query})
        rich.print(human_response)
        return human_response['data']

    def extract_tools(self, state: AgentState):
        messages = state.get('messages', [])
        if messages:
            prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=GET_TOOL_SYSTEM), MessagesPlaceholder(variable_name='history')]
            )
            chain = prompt | self.struct_llm.with_structured_output(ToolsOutput)
            output: ToolsOutput = chain.invoke({'history': state.get('messages')})

            return {'paper_tools': output}
        else:
            prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=GET_TOOL_SYSTEM), ('human', GET_TOOL_USER)]
            )
            chain = prompt | self.struct_llm.with_structured_output(ToolsOutput)
            output: ToolsOutput = chain.invoke({'paper_content': state.get('paper_content')})

            human_question = (
                ChatPromptTemplate.from_messages([('human', GET_TOOL_USER)])
                .invoke(state.get('paper_content'))
                .to_messages()
            )
            return {'paper_tools': output, 'messages': human_question}

    @staticmethod
    def analyse_extract_result(state: AgentState) -> Command[Literal['extract_tools', 'check_tool']]:
        paper_tools = state.get('paper_tools')

        if len(paper_tools.tools) == 0:
            question = 'I found 0 tools in the paper, please checkout the content you provide.'
            answer = interrupt(question)
            return Command(
                goto='extract_tools', update={'messages': [AIMessage(content=question), HumanMessage(content=answer)]}
            )

        if paper_tools.unsure_tools:
            question = (
                f'I found {len(paper_tools.unsure_tools) + len(paper_tools.tools)} tools in the paper, '
                f'but not sure about {len(paper_tools.unsure_tools)} of them: '
                f'{"、".join([t.name for t in paper_tools.unsure_tools])}. Please give your advice。'
            )
            answer = interrupt(question)
            return Command(
                goto='extract_tools', update={'messages': [AIMessage(content=question), HumanMessage(content=answer)]}
            )

        return Command(
            goto=[Send('check_tool', {'name': t.name, 'content': t.paper_text}) for t in state['paper_tools'].tools],
            update={
                'messages': [
                    AIMessage(
                        content=f'I found {len(paper_tools.tools)} tools in paper: [{", ".join([t.name for t in paper_tools.tools])}]'
                    )
                ]
            },
        )

    def check_tool(self, state: CheckState) -> Command[Literal['download_tool', 'add_tool', 'analyse_check_result']]:
        with Session(engine) as session:
            db_tools = tool_crud.get_by_name_fuzzy(session, state['name'], 20, 0)

        if len(db_tools) == 0:
            return Command(goto='download_tool')

        formatter = itemgetter('db_tools') | RunnableLambda(tool_formatter)
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=CHECK_DB_SYSTEM), ('human', CHECK_DB_USER)])
        chain = (
            RunnablePassthrough.assign(tool_info=formatter)
            | prompt
            | self.struct_llm.with_structured_output(DBChooseResult)
        )

        response: DBChooseResult = chain.invoke({'db_tools': db_tools, 'paper_text': state['content']})

        if response.is_right_tool:
            if len(response.missed_tools) == 0:
                return Command(goto='analyse_check_result')
            else:
                return Command(goto='add_tool')
        else:
            return Command(goto='download_tool')

    def download_tool(self, state: CheckState):
        pass

    def add_tool(self, state: CheckState):
        pass

    def analyse_check_result(self, state: AgentState):
        pass

    def build(self, memory: BaseCheckpointSaver, visualize: bool = False) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node('extract_tools', self.extract_tools)
        graph.add_node('analyse_extract_result', self.analyse_extract_result)
        graph.add_node('check_tool', self.check_tool)
        graph.add_node('download_tool', self.download_tool)
        graph.add_node('add_tool', self.add_tool)
        graph.add_node('analyse_check_result', self.analyse_check_result)

        graph.add_edge(START, 'extract_tools')
        graph.add_edge('extract_tools', 'analyse_extract_result')
        graph.add_edge('download_tool', 'add_tool')
        graph.add_edge('add_tool', 'check_tool')

        compiled_graph = graph.compile(checkpointer=memory)
        if visualize:
            compiled_graph.get_graph(xray=True).draw_mermaid_png(
                output_file_path='reproduction_agent.png',
                draw_method=MermaidDrawMethod.API,
                curve_style=CurveStyle.BASIS,
            )
        return compiled_graph
