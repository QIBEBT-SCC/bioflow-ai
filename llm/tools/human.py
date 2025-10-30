from langchain_core.tools import tool
from langgraph.types import interrupt


@tool(parse_docstring=True)
def ask_human(question: str):
    """Consult humans for advice

    Args:
        question: Issues to consult with humans
    """
    answer = interrupt({'question': question})
    return f'Human feedback is: "{answer}"'
