import operator
from typing import Literal, Annotated, Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import ToolException
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.core.llm_chat import gpt_41_mini
from app.llm.tools.container import RunContainer
from app.models.tools import ImageConfig

GLOBAL_SYSTEM_PROMPT = """You are an expert specializing in the analysis of bioinformatics tools. The user wants to obtain the help documentation for a specific tool.

<goal>  
Your task is to:  
1. Analyze the user's input to determine the correct command to run.  
2. Use the `run_command` tool to execute the appropriate command to retrieve help information.  
3. Analyze the command output to determine whether it includes subcommands.  
4. If subcommands exist, retrieve detailed help information for each of them as follows:  
   * For the **first** subcommand, invoke the tool separately and continue trying until you get a valid result.  
   * Once the first subcommand succeeds, proceed to invoke the tool **multiple times in parallel** to retrieve help information for the **remaining subcommands**.  
   * In particular, subcommands such as `help` and `version`, which are not used in the actual analysis, **do not** need to be analyzed.
5. When you believe enough help information has been collected, say **"I have finished collecting all the help information"** to indicate the end of the collection phase.  
</goal>

<tool_calling>
* The tool is already running in a Docker container. You only need to provide the bare command-line instructions — do **not** include prefixes like `docker run`.
* If the tool has subcommands, follow the above procedure to retrieve help documentation.
</tool_calling>

<error_handling>
If the tool returns information that is not valid software help documentation, it means your command was incorrect and should be revised.

* If you receive an error like `command not found`, it indicates the tool name is incorrect. Try a new, correct tool name based on your knowledge.
    * If you can't find the name of the tool in your knowledge, try to remember if you have to add a suffix such as `.py`, `.sh`, etc.
* If you receive an error like `unrecognized arguments`, it means the parameters are incorrect. In this case:
  * First, try to determine the correct command for displaying the help documentation based on the returned message.
  * If that fails, try the following options in order:
    * `tool_name --help` or `tool_name -h`
    * `tool_name help`
    * `tool_name` (some tools show help when run with no arguments)
</error_handling>

Please choose the most appropriate command based on the tool name and progressively retrieve the complete help documentation.
"""

ERROR_SYS_PROMPT = """You are an expert specializing in the analysis of bioinformatics tools. The user wants to obtain the help documentation for a specific tool.

Now, we will attempt to retrieve the help documentation for a subcommand of a tool. 
Your task is to determine based on the system's returned result whether the content matches the expected subcommand help information. 
You will receive the tool's main help message, the command executed this time, and the return value of the subcommand.
"""

ERROR_USER_PROMPT = """<main_doc>
{main_doc}
</main_doc>

Now we try the command:
```
{command}
```

and the return is:
<sub_command_return>
{sub_command_return}
</sub_command_return>
"""


class HelpDocument(BaseModel):
    command: str
    description: Optional[str] = None
    content: str


class ErrorStructure(BaseModel):
    """Is the output obtained here the expected help document for the subcommand?"""

    is_doc: bool = Field(description='Is the returned text a help document for a subcommand?')


class HelpDocumentState(MessagesState):
    """智能体状态管理"""

    image_config: ImageConfig

    documents: Annotated[list[HelpDocument], operator.add]
    tool_name: str


class HelpDocumentOutput(BaseModel):
    message: AIMessage
    documents: list[HelpDocument]
    tool_name: str


class HelpDocumentAgent:
    """智能帮助文档获取智能体"""

    def __init__(self):
        self.llm = gpt_41_mini()
        self.structure_llm = gpt_41_mini()

        self.container_tool = RunContainer()

    def call_llm_node(self, state: HelpDocumentState):
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=GLOBAL_SYSTEM_PROMPT), MessagesPlaceholder(variable_name='history')]
        )
        llm_with_tools = self.llm.bind_tools([self.container_tool], tool_choice='auto', parallel_tool_calls=True)
        chain = prompt | llm_with_tools
        response = chain.invoke({'history': state['messages']})

        return {'messages': [response]}

    @staticmethod
    def should_continue(
        state: HelpDocumentState,
    ) -> Literal['run_command_node', 'run_subcommand_node', 'parallel_run_node', 'analyse_document_node']:
        last_message = state['messages'][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if len(state.get('documents', [])) == 0:
                return 'run_command_node'

            if len(last_message.tool_calls) > 1:
                return 'parallel_run_node'

            return 'run_subcommand_node'

        return 'analyse_document_node'

    def run_command_node(self, state: HelpDocumentState):
        last_message = state['messages'][-1]
        tool_call: ToolCall = last_message.tool_calls[0]
        tool_call_id = tool_call['id']

        try:
            command_output = self.container_tool.invoke({**tool_call['args'], 'image_config': state['image_config']})
            return {
                'messages': [ToolMessage(command_output, tool_call_id=tool_call_id)],
                'documents': [HelpDocument(command=tool_call['args']['command'], content=command_output)],
            }
        except ToolException as e:
            return {'messages': [ToolMessage(str(e), tool_call_id=tool_call_id)]}

    def run_subcommand_node(self, state: HelpDocumentState):
        last_message = state['messages'][-1]
        tool_call: ToolCall = last_message.tool_calls[0]
        tool_call_id = tool_call['id']

        try:
            command_output = self.container_tool.invoke({**tool_call['args'], 'image_config': state['image_config']})
        except ToolException as e:
            command_output = str(e)

        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=ERROR_SYS_PROMPT), ('human', ERROR_USER_PROMPT)]
        )
        chain = prompt | self.structure_llm.with_structured_output(ErrorStructure)
        response: ErrorStructure = chain.invoke(
            {
                'main_doc': state.get('documents')[0].content,
                'command': tool_call['args'],
                'sub_command_return': command_output,
            }
        )

        if response.is_doc:
            return {
                'messages': [
                    ToolMessage(
                        'Successful get the subcommand help doc, add to documents. Now you can call the tool for the remaining subcommands in parallel.',
                        tool_call_id=tool_call_id,
                    )
                ],
                'documents': [
                    HelpDocument(
                        command=tool_call['args']['command'],
                        content=command_output,
                        description=tool_call['args']['command_description'],
                    )
                ],
            }

        return {'messages': [ToolMessage(command_output, tool_call_id=tool_call_id)]}

    def parallel_run_node(self, state: HelpDocumentState):
        last_messages = state.get('messages', [])[-1]
        tool_messages = []
        new_documents = []

        for tool_call in last_messages.tool_calls:
            try:
                command_output = self.container_tool.invoke(
                    {**tool_call['args'], 'image_config': state.get('image_config')}
                )
            except ToolException as e:
                command_output = str(e)

            new_documents.append(
                HelpDocument(
                    command=tool_call['args']['command'],
                    content=command_output,
                    description=tool_call['args']['command_description'],
                )
            )
            tool_messages.append(
                ToolMessage('Successful get the subcommand help doc, add to documents', tool_call_id=tool_call['id'])
            )

        return {'messages': tool_messages, 'documents': new_documents}

    @staticmethod
    def analyse_document_node(state: HelpDocumentState):
        """格式化最终文档"""
        documents = state.get('documents', [])

        if not documents:
            return {'message': AIMessage(content='我无法获取到有效的帮助文档，请检查相关的信息是否正确。')}

        tool_name = documents[0].command.split(' ')[0]
        return {
            'message': AIMessage(content=f'已生成完整的帮助文档，包含 {len(documents)} 个部分。'),
            'tool_name': tool_name,
        }

    def build(self, visualize: bool = False) -> CompiledStateGraph:
        graph = StateGraph(HelpDocumentState, output=HelpDocumentOutput)

        graph.add_node('call_llm_node', self.call_llm_node)
        graph.add_node('run_command_node', self.run_command_node)
        graph.add_node('run_subcommand_node', self.run_subcommand_node)
        graph.add_node('parallel_run_node', self.parallel_run_node)
        graph.add_node('analyse_document_node', self.analyse_document_node)

        graph.set_entry_point('call_llm_node')
        graph.add_conditional_edges('call_llm_node', self.should_continue)
        graph.add_edge('run_command_node', 'call_llm_node')
        graph.add_edge('run_subcommand_node', 'call_llm_node')
        graph.add_edge('parallel_run_node', 'analyse_document_node')
        graph.add_edge('analyse_document_node', END)

        compiled_graph = graph.compile()
        if visualize:
            compiled_graph.get_graph(xray=True).draw_mermaid_png(
                output_file_path='help_doc_agent.png', draw_method=MermaidDrawMethod.API, curve_style=CurveStyle.BASIS
            )
        return compiled_graph
