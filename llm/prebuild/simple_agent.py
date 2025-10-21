from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode

from llm.models import gemini_25_flash
from llm.tools.web_search import web_reader, web_searcher

summarization_model = gemini_25_flash().bind(max_tokens=2000)

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=13000,
    max_summary_tokens=2000,
    output_messages_key='llm_input_messages',
)


class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]


web_search_agent = create_react_agent(
    model=gemini_25_flash(),
    tools=[web_searcher, web_reader],
    name='simple_chat',
    state_schema=State,
    prompt=(
        'You are a general-purpose conversational assistant that can call relevant tools for web searches to aid in answering when necessary. '
    ),
    pre_model_hook=summarization_model,
)
