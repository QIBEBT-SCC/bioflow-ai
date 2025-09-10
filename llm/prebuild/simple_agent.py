from langgraph.prebuilt import create_react_agent

from llm.models import gemini_25_flash
from llm.tools.web_search import jina_search, jina_reader

web_search_agent = create_react_agent(
    model=gemini_25_flash(),
    tools=[jina_search, jina_reader],
    name='web_searcher',
    prompt='You are a web information retrieval expert, using tools to retrieve and read web information to answer questions.',
)
