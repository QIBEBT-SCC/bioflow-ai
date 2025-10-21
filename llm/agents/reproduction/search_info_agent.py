from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from pydantic import BaseModel, Field

from llm.models import gemini_25_flash
from llm.tools.web_search import conclude_web_reader_result, web_reader, web_searcher

SEARCH_SYS = """To organize information about bioinformatics tools, we need to search for their corresponding GitHub repositories and paper links. 
To search for this information, you need to call the `web_search` tool to search online for relevant information. 

For papers, you can use the following steps to query:
- First, try calling the `web_reader` tool to directly read the tool's GitHub page to see if there is a description of references on the homepage.
- If not, try searching online. If both preprints and published literature are found, prioritize published literature.
- If there are multiple relevant papers, prioritize the earliest published, original paper for that tool.
- If none are found, the paper link can be empty.

When you are sure you have enough information, do not call the tool again, return the result, and the task ends."""


class OutputStructure(BaseModel):
    github_url: str = Field(description='GitHub repository link for the tool')
    paper_url: Optional[str] = Field(description="Link to the tool's publication, leave blank if not applicable")


class AgentState(AgentStatePydantic):
    final_response: Optional[OutputStructure] = None


class SearchInfoAgent:
    def __init__(self):
        self.llm = gemini_25_flash(0.3)

        self.tools = [web_searcher, web_reader, conclude_web_reader_result]
        self.tool_node = ToolNode(self.tools)

    async def call_model_node(self, state: AgentState):
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=SEARCH_SYS), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.bind_tools(self.tools)
        response = await chain.ainvoke({'history': state.messages})

        return {'messages': [response]}

    async def respond_node(self, state: AgentState):
        prompt = ChatPromptTemplate(
            messages=[SystemMessage(content=SEARCH_SYS), MessagesPlaceholder(variable_name='history')]
        )
        chain = prompt | self.llm.with_structured_output(OutputStructure)
        response = await chain.ainvoke({'history': state.messages[:-1]})

        return {'final_response': response}

    @staticmethod
    def should_continue(state: AgentState):
        last_message = state.messages[-1]

        if not last_message.tool_calls:
            return 'respond'
        else:
            return 'continue'

    def build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node('agent', self.call_model_node)
        workflow.add_node('respond', self.respond_node)
        workflow.add_node('tools', self.tool_node)

        workflow.set_entry_point('agent')
        workflow.add_conditional_edges('agent', self.should_continue, {'continue': 'tools', 'respond': 'respond'})

        workflow.add_edge('tools', 'agent')
        workflow.add_edge('respond', END)
        graph = workflow.compile()

        return graph
