# nodes/agent.py

from langchain_core.messages import SystemMessage
from llm.llm_client import get_llm
from llm.prompts import AGENT_SYSTEM_PROMPT
from tools.vector_search_tool import vector_search
from tools.web_search_tool import web_search
from graph.state import AgentState


TOOLS = [vector_search, web_search]

# Build once at import time — not on every agent loop iteration
_llm_with_tools = None

def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    llm_with_tools = _get_llm_with_tools()

    messages = list(state["messages"])

    # Prepend system prompt only once, at the very start
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}