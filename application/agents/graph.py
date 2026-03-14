import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
import langchainhub as hub
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from adapters.tools.calculator import calculator
from adapters.tools.retrieval_tool import retrieval_tool

# 1. Define the state
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Define tools and the LLM
tools = [calculator, retrieval_tool]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Define the nodes
def agent_node(state: MessagesState):
    messages = state["messages"]
    
    try:
        # Dynamically pull the prompt from LangSmith hub
        # Format: username/repo-name or hwchase17/react etc. 
        # Here we use the exact prompt name created in LangSmith
        prompt = hub.pull("agentic-rag-prompt")
        system_message = SystemMessage(content=prompt.template)
    except Exception:
        # Fallback system prompt if LangSmith hub pull fails (e.g. not configured yet)
        system_message = SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "Evaluate the user's question explicitly:\n"
                "1. If the question requires external knowledge about RAG, LangGraph, pgvector, or FastAPI, "
                "use the `retrieval_tool`.\n"
                "2. Otherwise, answer directly or use other available tools (like calculator) if appropriate."
            )
        )

    # The LLM looks at the conversation history and decides whether to yield a tool call or a final answer
    response = llm_with_tools.invoke([system_message] + messages)
    return {"messages": [response]}

# Define the function that determines whether to continue to the tool node or end
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue to the tools
    return "continue"

# 4. Define the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.set_entry_point("agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
builder.add_edge("tools", "agent")

# Compile the graph
graph = builder.compile()
