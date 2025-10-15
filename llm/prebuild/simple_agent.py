from langgraph.prebuilt import create_react_agent

from llm.models import gemini_25_flash
from llm.tools.web_search import conclude_web_reader_result, web_reader, web_searcher

web_search_agent = create_react_agent(
    model=gemini_25_flash(),
    tools=[web_searcher, web_reader, conclude_web_reader_result],
    name='web_searcher',
    prompt='You are a web information retrieval expert, using tools to retrieve and read web information to answer questions. After each web page is read, `conclude_web_reader_result` needs to be called to compress the context.',
)
