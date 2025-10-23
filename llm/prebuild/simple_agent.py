from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from llm.models import gemini_25_flash
from llm.tools.web_search import web_reader, web_searcher

summarization_model = gemini_25_flash()

web_search_agent = create_agent(
    model=gemini_25_flash(),
    tools=[web_searcher, web_reader],
    system_prompt='You are a general-purpose conversational assistant that can call relevant tools for web searches to aid in answering when necessary.',
    middleware=[
        SummarizationMiddleware(
            model=summarization_model,
            max_tokens_before_summary=6000,
            messages_to_keep=6,
        )
    ],
    name='simple_chat',
)
