from research_analyst.state import ResearchState


def re_search_plan_node(state: ResearchState) -> dict:
    """
    Refines the search query based on the critic's feedback and
    increments the retry count. Routes back to tool_executor.
    """
    critique = state.get("critique_result", {})
    original_query = state["query"]
    retry_count = state.get("retry_count", 0)

    # Use the critic's suggested refinement if available, else fall back to original
    refined_query = critique.get("suggested_refinement") or original_query
    missing = critique.get("missing_aspects", [])

    # If there are specific missing aspects, append them to sharpen the query
    if missing:
        aspects_str = ", ".join(missing)
        refined_query = f"{refined_query} Focus on: {aspects_str}"

    # On retry, add web_search as a fallback tool if not already included
    tools_to_use = list(state.get("tools_to_use", ["rag", "finance"]))
    if "web_search" not in tools_to_use:
        tools_to_use.append("web_search")

    return {
        "current_query": refined_query,
        "tools_to_use": tools_to_use,
        "retry_count": retry_count + 1,
    }
