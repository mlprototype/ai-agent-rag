from langchain_core.tools import tool

@tool
def calculator(a: int, b: int) -> int:
    """Add two numbers together. Use this tool when you need to calculate the sum of two integers."""
    return a + b
