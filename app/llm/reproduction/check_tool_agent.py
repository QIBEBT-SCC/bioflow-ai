import operator
from enum import IntEnum
from typing import TypedDict, Literal, Optional, Annotated

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import ToolException
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Command, Send
from loguru import logger
from pydantic import BaseModel, Field
from sqlmodel import Session

import app.crud.tools.tool_doc as doc_crud
from app.core.llm_chat import gpt_41_mini
from app.core.session import engine
from app.llm.reproduction.add_tool_agent import AddToolAgent, AddToolState
from app.llm.reproduction.get_help_agent import HelpDocumentAgent, HelpDocument
from app.llm.tools.images import SearchImage
from app.models.tools import ImageConfig, ToolHelpDoc

SELECT_DOC_SYS = """You are an intelligent assistant designed to select the most suitable tool(s) and subcommand(s) from provided help documentation to fulfill a task described in an original text snippet.
You must analyze the original text's description of a tool's function and match it with the capabilities described in the provided help documentation.

<Task_Description>
Your goal is to identify one or more specific subcommands (and potentially the main command) from the help documentation that can perform the function described in the original content. If a subcommand is selected, you must also define a precise task for it to accomplish.
</Task_Description>

<Input_Format>
The user will provide two pieces of information:
1.  **"The origin content of tool"**: This is a text snippet from a document describing a tool and its function. This is your primary source for understanding the user's need.
2.  **"help documentation for the existing tools"**: This is a structured document formatted in XML-like tags, containing one or more `<tool_help_doc>` blocks. Each block represents a main tool and may contain `<sub_commands>` with multiple `<sub_command>` entries.
    * `<doc_id>`: The unique ID for the main tool.
    * `<help_command>`: The command name for the tool.
    * `<main_doc>` or `<help_doc>`: A description of the main tool's overall function.
    * `<sub_command>`: A block for each subcommand.
        * `<sub_doc_id>`: The unique ID for the subcommand.
        * `<sub_help_command>`: The command name for the subcommand.
        * `<subcommand_description>`: A description of the subcommand's specific function.
</Input_Format>

<Output_Format>
You must output a JSON array, where each element is an object with two fields:
* `id`: The integer ID of the selected subcommand.
* `task`: A string describing the specific task that needs to be performed using this subcommand to achieve the functionality mentioned in the original content.

If no suitable tool or subcommand is found, return an empty JSON array `[]`.

<Constraints>
1.  You must first determine the most relevant main tool based on the original content. If multiple main tools are provided, select the one that is the best match.
2.  If the selected main tool has subcommands, your primary focus should be on selecting one or more subcommands that directly fulfill the described function.
3.  If a main tool has no subcommands, you should select the main tool's ID. In this case, the `task` field should describe what the main tool needs to do.
4.  The `task` field must be a clear, actionable instruction for the selected tool/subcommand. This instruction should focus on the data to be processed (input data type, format, source), the expected output (output file type, format), and any relevant parameter settings mentioned or implied in the original content.
5.  Base your selection **solely** on the content of "The origin content of tool" and the provided "help documentation". Do not use external knowledge.
6.  The output must be a valid JSON array and nothing else. Do not include any explanatory text, markdown formatting (e.g., code blocks), or conversational phrases outside of the JSON output.
</Constraints>

<Thinking_Process>
1.  **Analyze the "origin content"**: Read the original text snippet to understand the functionality and purpose of the tool being described. Identify key actions, objects, goals, input data types, output formats, and specific parameters.
2.  **Scan "help documentation"**: Go through each `<tool_help_doc>` block.
3.  **Match Main Tools**: Compare the overall description of each main tool (`<main_doc>` or `<help_doc>`) with the original content. Select the most relevant main tool. If a tool name is mentioned in the original content, prioritize matching that name.
4.  **Evaluate Subcommands**: If the selected main tool has subcommands, analyze each `<sub_command>`. Compare the `<subcommand_description>` of each subcommand with the functionality needed from the original content.
5.  **Select IDs and Define Tasks**: Choose the `sub_doc_id` or `doc_id` that best matches the required function. For each selected ID, formulate a specific `task` that the tool needs to perform.
6.  **Construct JSON**: Format the final selection(s) as a JSON array of objects, each containing an `id` and a `task`.
</Thinking_Process>
"""

SELECT_DOC_USER = """The origin content of tool:
{content}

Below is the help documentation for the existing tools searched based on the tool name:
{doc}
"""


def doc_formatter(main_doc: ToolHelpDoc, sub_docs: Optional[list[ToolHelpDoc]] = None) -> str:
    if sub_docs:
        sub_commands = '\n'.join(
            [
                f'        <sub_command>\n'
                f'            <sub_doc_id>{sb.id}</sub_doc_id>\n'
                f'            <sub_help_command>{sb.help_command}</sub_help_command>\n'
                f'            <subcommand_description>{sb.description}</subcommand_description>\n'
                f'        </sub_command>'
                for sb in sub_docs
            ]
        )
        doc = (
            f'<tool_help_doc>\n'
            f'    <doc_id>{main_doc.id}</doc_id>\n'
            f'    <help_command>{main_doc.help_command}</help_command>\n'
            f'    <main_doc>\n'
            f'    {main_doc.content}\n'
            f'    </main_doc>\n'
            f'    <sub_commands>\n'
            f'{sub_commands}\n'
            f'    </sub_commands>\n'
            f'</tool_help_doc>'
        )
    else:
        doc = (
            f'<tool_help_doc>\n'
            f'    <doc_id>{main_doc.id}</doc_id>\n'
            f'    <help_command>{main_doc.help_command}</help_command>\n'
            f'    <help_doc>\n'
            f'    {main_doc.content}\n'
            f'    </help_doc>\n'
            f'</tool_help_doc>'
        )
    return doc


class DocSelectResult(BaseModel):
    """command select result"""

    id: int = Field(description='selected sub command id')
    task: str = Field(
        description='In order to realize the described task, you need the specific task that this tool (sub-tool) realizes'
    )


class DocSelectResults(BaseModel):
    results: list[DocSelectResult] = Field('One or more tool(s) you selected')


class MissingTool(BaseModel):
    help_doc: ToolHelpDoc
    task: str


class ToolErrorEnum(IntEnum):
    IMAGE_NOT_FOUND = 0
    HELP_COMMAND_ERROR = 1


class ToolErrorInfo(BaseModel):
    name: str
    content: str
    reason: int


class CheckState(TypedDict):
    name: str
    content: str
    image: ImageConfig

    documents: str

    exist_tools: Annotated[list[str], operator.add]
    missing_tools: Annotated[list[MissingTool], operator.add]


class CheckToolAgent:
    def __init__(self):
        self.llm = gpt_41_mini()
        self.structure_llm = gpt_41_mini()

        self.image_search_tool = SearchImage()
        self.get_help_agent = HelpDocumentAgent().build()
        self.add_tool_agent = AddToolAgent().build()

    @staticmethod
    def search_document_node(state: CheckState) -> Command[Literal['download_node', 'check_doc_node']]:
        # TODO 这里要确保进来的名字一定是主工具名，在前面的名称提取里面加约束
        with Session(engine) as session:
            tool_docs: list[ToolHelpDoc] = doc_crud.get_by_name_fuzzy(session, state['name'], 50, 0)
            if len(tool_docs) == 0:
                return Command(goto='download_node')

            doc_list = []
            for main_doc in tool_docs:
                parent_id = main_doc.id
                sub_docs = doc_crud.get_sub_docs(session, parent_id)

                if sub_docs:
                    doc_list.append(doc_formatter(main_doc, sub_docs))
                else:
                    # 我们假定所有工具在初始化的时候均已经生成了完整的文档
                    # 没有子命令说明只有单个命令
                    doc_list.append(doc_formatter(main_doc))

            tool_name = main_doc.help_command.split(' ')[0]

        return Command(goto='check_doc_node', update={'name': tool_name, 'documents': '\n\n'.join(doc_list)})

    def download_node(self, state: CheckState) -> Command[Literal['generate_help_node', '__end__']]:
        try:
            image_cfg = self.image_search_tool.invoke({'tool_name': state['name'], 'origin_text': state['content']})
            logger.debug(image_cfg)

            return Command(goto='generate_help_node', update={'image': image_cfg})
        except ToolException as e:
            logger.warning(e)
            return Command(
                goto='__end__',
                update={
                    'check_result': [
                        ToolErrorInfo(
                            name=state['name'], content=state['content'], reason=ToolErrorEnum.IMAGE_NOT_FOUND
                        )
                    ]
                },
            )

    def generate_help_node(self, state: CheckState) -> Command[Literal['check_doc_node', '__end__']]:
        response = self.get_help_agent.invoke(
            {'messages': f'Get help document for {state["name"]}', 'image_config': state['image']}
        )

        if response['documents']:
            main_doc: HelpDocument = response['documents'].pop(0)
            db_main_doc = ToolHelpDoc(help_command=main_doc.command, content=main_doc.content, image=state['image'])

            with Session(engine) as session:
                db_main_doc = doc_crud.insert(session, db_main_doc)
                sub_docs = None
                if response['documents']:
                    sub_docs = [
                        ToolHelpDoc(
                            help_command=sub_doc.command,
                            content=sub_doc.content,
                            description=sub_doc.description,
                            image=state['image'],
                            parent_id=db_main_doc.id,
                        )
                        for sub_doc in response['documents']
                    ]

                    sub_docs = doc_crud.insert_all(session, sub_docs)

                doc_text = doc_formatter(db_main_doc, sub_docs)
            return Command(goto='check_doc_node', update={'name': response['tool_name'], 'documents': doc_text})
        else:
            return Command(
                goto='__end__',
                update={
                    'check_result': [
                        ToolErrorInfo(
                            name=state['name'], content=state['content'], reason=ToolErrorEnum.HELP_COMMAND_ERROR
                        )
                    ]
                },
            )

    def check_doc_node(self, state: CheckState) -> Command[Literal['add_tool_node', '__end__']]:
        prompt = ChatPromptTemplate(messages=[SystemMessage(content=SELECT_DOC_SYS), ('human', SELECT_DOC_USER)])
        chain = prompt | self.structure_llm.with_structured_output(DocSelectResults)

        response: DocSelectResults = chain.invoke({'content': state['content'], 'doc': state['documents']})

        if response.results:
            # 到这一步，已经可以确定给出的命令都是真实存在的
            with Session(engine) as session:
                for result in response.results:
                    doc: ToolHelpDoc = doc_crud.get(session, result.id)

                    if 'image' not in state:
                        state['image'] = doc.image

                    if doc.tools:
                        state['exist_tools'].extend([t.uid for t in doc.tools])
                    else:
                        state['missing_tools'].append(MissingTool(help_doc=doc, task=result.task))

            if state['missing_tools']:
                return Command(
                    goto=[
                        Send(
                            'add_tool_node',
                            {
                                'image': state['image'],
                                'tool_name': state['name'],
                                'help_doc_id': doc.id,
                                'help_doc': f'{doc.help_command}\n{doc.content}',
                                'origin_query': t.task,
                            },
                        )
                        for t in state['missing_tools']
                    ]
                )
            return Command(goto='__end__')

        return Command(
            goto='__end__',
            update={
                'check_result': [
                    ToolErrorInfo(name=state['name'], content=state['content'], reason=ToolErrorEnum.HELP_COMMAND_ERROR)
                ]
            },
        )

    def add_tool_node(self, sub_state: AddToolState):
        agent = self.add_tool_agent
        response = agent.invoke(
            {
                'image': sub_state['image'],
                'tool_name': sub_state['tool_name'],
                'help_doc': sub_state['help_doc'],
                'help_doc_id': sub_state['help_doc_id'],
                'origin_query': sub_state['origin_query'],
            }
        )
        return {'exist_tools': [sub_state['origin_query']]}

    def build(self):
        graph = StateGraph(CheckState)

        graph.add_node('search_document_node', self.search_document_node)
        graph.add_node('download_node', self.download_node)
        graph.add_node('generate_help_node', self.generate_help_node)
        graph.add_node('check_doc_node', self.check_doc_node)
        graph.add_node('add_tool_node', self.add_tool_node)

        graph.set_entry_point('search_document_node')
        graph.add_edge('add_tool_node', END)

        return graph.compile()

    def for_visual(self):
        graph = StateGraph(CheckState)

        graph.add_node('search_document_node', self.search_document_node)
        graph.add_node('download_node', self.download_node)
        graph.add_node('generate_help_node', self.get_help_agent)
        graph.add_node('check_doc_node', self.check_doc_node)
        graph.add_node('add_tool_node', self.add_tool_agent)

        graph.set_entry_point('search_document_node')
        graph.add_conditional_edges('generate_help_node', lambda x: x, ['check_doc_node', END])
        graph.add_edge('add_tool_node', END)

        compiled_graph = graph.compile()
        compiled_graph.get_graph(xray=True).draw_mermaid_png(
            output_file_path='check_tool_agent.png',
            draw_method=MermaidDrawMethod.API,
            curve_style=CurveStyle.BASIS,
            frontmatter_config={'config': {'theme': 'redux', 'look': 'neo'}},
        )
        return compiled_graph
