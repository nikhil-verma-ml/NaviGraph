from graph.state import AgentState


def should_continue(state: AgentState) -> str:
    """
    Checks the last message in the state.
    If the LLM requested a tool call, route to the tool node.
    Otherwise, the LLM has given a final answer, so end the graph.
    """
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"