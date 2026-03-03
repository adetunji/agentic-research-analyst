from langgraph.graph import StateGraph, END

from research_analyst.state import ResearchState
from research_analyst.config import settings
from research_analyst.nodes.planner import planner_node
from research_analyst.nodes.tool_executor import tool_executor_node
from research_analyst.nodes.critic import critic_node
from research_analyst.nodes.re_search_plan import re_search_plan_node
from research_analyst.nodes.synthesizer import synthesizer_node
from research_analyst.nodes.formatter import formatter_node


def route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge after the critic node.
    - PASS: relevance score >= threshold and has_sufficient_context → synthesizer
    - FAIL: retry budget remaining → re_search_plan
    - Force forward: retry budget exhausted → synthesizer regardless
    """
    retry_count = state.get("retry_count", 0)
    critique = state.get("critique_result", {})

    if retry_count >= settings.max_critique_retries:
        return "synthesizer"

    if critique.get("has_sufficient_context", False):
        return "synthesizer"

    return "re_search_plan"


def build_graph():
    """
    Assembles and compiles the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(ResearchState)

    # --- Register nodes ---
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("critic", critic_node)
    graph.add_node("re_search_plan", re_search_plan_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("formatter", formatter_node)

    # --- Define edges ---
    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_edge("tool_executor", "critic")

    # Conditional branch: PASS → synthesizer, FAIL → re_search_plan
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer",
            "re_search_plan": "re_search_plan",
        },
    )

    # Retry loop: re_search_plan routes back to tool_executor
    graph.add_edge("re_search_plan", "tool_executor")

    graph.add_edge("synthesizer", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
