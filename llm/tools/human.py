from langchain_core.tools import tool
from langgraph.types import interrupt


@tool('ask_human', parse_docstring=True)
def ask_human(question: str) -> str:
    """Consult humans for advice.

    Args:
        question: Issues to consult with humans
    """
    answer = interrupt({'question': question})
    return f'Human feedback is: "{answer}"'
