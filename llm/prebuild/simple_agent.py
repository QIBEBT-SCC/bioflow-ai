from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from llm.models import gemini_25_flash, gpt_41_mini


@tool
def add(a: int, b: int) -> int:
    """计算两个数的和

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def multiple(a: int, b: int) -> int:
    """计算两个数相乘的结果

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


simple_agent = create_react_agent(model=gpt_41_mini(), tools=[add, multiple])
