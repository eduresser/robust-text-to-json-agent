from langgraph.graph import END, StateGraph

from text_to_json.agent.nodes import (
    call_llm_node,
    chunk_text_node,
    execute_tools_node,
    finalize_chunk_node,
    has_more_chunks,
    is_chunk_done,
    prepare_chunk_node,
)
from text_to_json.agent.state import AgentState


def create_graph() -> StateGraph:
    """
    Create the LangGraph for structured data extraction.

    Returns:
        The compiled graph ready to execute.
    """
    graph = StateGraph(AgentState)

    graph.add_node("chunk_text", chunk_text_node)
    graph.add_node("prepare_chunk", prepare_chunk_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("finalize_chunk", finalize_chunk_node)

    graph.set_entry_point("chunk_text")

    graph.add_edge("chunk_text", "prepare_chunk")

    graph.add_conditional_edges(
        "prepare_chunk",
        has_more_chunks,
        {
            "call_llm": "call_llm",
            "__end__": END,
        },
    )

    graph.add_edge("call_llm", "execute_tools")

    graph.add_conditional_edges(
        "execute_tools",
        is_chunk_done,
        {
            "finalize_chunk": "finalize_chunk",
            "call_llm": "call_llm",
        },
    )

    graph.add_edge("finalize_chunk", "prepare_chunk")

    return graph.compile()
