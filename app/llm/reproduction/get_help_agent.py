import operator
from typing import Literal, Annotated, Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle
from langchain_core.tools import ToolException
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
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
* The `command_description` field should describe the function or purpose of the subcommand itself, not the action of retrieving its help documentation.
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

<subcommand_handling>
**Important**: If you encounter multiple consecutive failures when trying to get help for subcommands (e.g., 3-4 failed attempts), this likely indicates that the "subcommands" shown in the main help are actually just usage patterns or parameter examples, not real subcommands. In such cases:
- Stop trying to get subcommand help
- Consider the main help document as the complete documentation
- Say **"I have finished collecting all the help information"** to end the collection phase
</subcommand_handling>

Please choose the most appropriate command based on the tool name and progressively retrieve the complete help documentation.
"""

ERROR_SYS = """You are an expert specializing in the analysis of bioinformatics tools. The user wants to obtain the help documentation for a specific tool.

## Scenario Description
When attempting to programmatically obtain help documentation for a tool, we've encountered cases where some tools do not provide direct help parameters (like `--help` or `-h`). Instead, they output the actual help documentation content as an **error message**. This makes it difficult to distinguish between genuine program errors and the desired help documentation.
Additionally, some tools may show usage patterns in their main help that look like subcommands, but these are not actual subcommands that can be called separately. When we try to get help for these "subcommands", we often get the same content as the main help document.

## Your Task
Please analyze the text provided by the user. Your goal is to determine if the text is the help documentation for a tool.

## Key Judgment Criteria
### For Main Command Help (when you see "Now we try the command:")
- Return `true` if the text contains help documentation for the tool, regardless of whether it's presented as an error message or not.
- Return `false` only if the text is clearly an error with no useful help information.

### For Subcommand Help (when you see "We've got the main help file for the tool"):
- Return `true` only if the text contains **NEW** help documentation specific to the subcommand that is different from the main help document.
- Return `false` if:
    - The subcommand return content is essentially the same as the main help document
    - The text is clearly an error with no useful help information
"""

SUB_ERROR_USER = """We've got the main help file for the tool
<main_doc>
{main_doc}
</main_doc>

Now we're trying to use this command to get the help documentation for the subcommand:
```
{command}
```

and the return is:
<sub_command_return>
{command_return}
</sub_command_return>
"""

MAIN_ERROR_USER = """Now we try the command:
```
{command}
```

and the return is:
<command_return>
{command_return}
</command_return>"""


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
    ) -> Literal['run_command_node', 'parallel_run_node', 'analyse_document_node']:
        last_message = state['messages'][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if len(last_message.tool_calls) > 1:
                return 'parallel_run_node'
            return 'run_command_node'

        return 'analyse_document_node'

    def run_command_node(self, state: HelpDocumentState) -> Command[Literal['analyse_output_node', 'call_llm_node']]:
        last_message = state['messages'][-1]
        tool_call: ToolCall = last_message.tool_calls[0]
        tool_call_id = tool_call['id']

        try:
            # 如果输出正常结束，肯定是一个有效指令，直接进下一步
            command_output = self.container_tool.invoke({**tool_call['args'], 'image_config': state['image_config']})

            if state['documents']:
                # 子文档
                tool_msg = ToolMessage(
                    content='Successful get the subcommand help doc, add to documents. Now you can call the tool for the remaining subcommands in parallel.',
                    tool_call_id=tool_call_id,
                )
            else:
                tool_msg = ToolMessage(command_output, tool_call_id=tool_call_id)
            return Command(
                goto='call_llm_node',
                update={
                    'messages': [tool_msg],
                    'documents': [HelpDocument(command=tool_call['args']['command'], content=command_output)],
                },
            )
        except ToolException as e:
            # 如果容器以非0状态退出，说明参数不正确。需要进一步判断
            return Command(
                goto='analyse_output_node', update={'messages': [ToolMessage(str(e), tool_call_id=tool_call_id)]}
            )

    def analyse_output_node(self, state: HelpDocumentState):
        ai_message = state['messages'][-2]
        tool_message = state['messages'][-1]
        assert isinstance(ai_message, AIMessage)
        assert isinstance(tool_message, ToolMessage)

        command = ai_message.tool_calls[0]['args']['command']

        if state['documents']:
            # 已经有帮助文档了，说明是在处理子命令
            prompt = ChatPromptTemplate.from_messages([SystemMessage(content=ERROR_SYS), ('human', SUB_ERROR_USER)])
            chain = prompt | self.structure_llm.with_structured_output(ErrorStructure)
            response: ErrorStructure = chain.invoke(
                {'main_doc': state['documents'][0].content, 'command': command, 'command_return': tool_message.content}
            )

            if response.is_doc:
                state['messages'].pop(-1)
                return {
                    'messages': [
                        ToolMessage(
                            content='Successful get the subcommand help doc, add to documents. Now you can call the tool for the remaining subcommands in parallel.',
                            tool_call_id=ai_message.tool_calls[0]['id'],
                        )
                    ],
                    'documents': [
                        HelpDocument(
                            command=command,
                            content=tool_message.content,
                            description=ai_message.tool_calls[0]['args']['command_description'],
                        )
                    ],
                }
        else:
            # 否则是本体帮助文档
            prompt = ChatPromptTemplate.from_messages([SystemMessage(content=ERROR_SYS), ('human', MAIN_ERROR_USER)])
            chain = prompt | self.structure_llm.with_structured_output(ErrorStructure)
            response: ErrorStructure = chain.invoke(
                {'command': ai_message.tool_calls[0]['args']['command'], 'command_return': tool_message.content}
            )

            if response.is_doc:
                return {
                    'documents': [
                        HelpDocument(command=ai_message.tool_calls[0]['args']['command'], content=tool_message.content)
                    ]
                }

        return {}

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
        graph = StateGraph(HelpDocumentState, output_schema=HelpDocumentOutput)  # type: ignore

        graph.add_node('call_llm_node', self.call_llm_node)  # type: ignore
        graph.add_node('run_command_node', self.run_command_node)  # type: ignore
        graph.add_node('analyse_output_node', self.analyse_output_node)  # type: ignore
        graph.add_node('parallel_run_node', self.parallel_run_node)  # type: ignore
        graph.add_node('analyse_document_node', self.analyse_document_node)  # type: ignore

        graph.set_entry_point('call_llm_node')
        graph.add_conditional_edges('call_llm_node', self.should_continue)
        graph.add_edge('analyse_output_node', 'call_llm_node')
        graph.add_edge('parallel_run_node', 'analyse_document_node')
        graph.add_edge('analyse_document_node', END)

        compiled_graph = graph.compile()
        if visualize:
            compiled_graph.get_graph(xray=True).draw_mermaid_png(
                output_file_path='help_doc_agent.png', draw_method=MermaidDrawMethod.API, curve_style=CurveStyle.BASIS
            )
        return compiled_graph
