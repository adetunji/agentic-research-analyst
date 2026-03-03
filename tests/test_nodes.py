import pytest
from unittest.mock import patch, MagicMock


def _mock_llm(content: str) -> MagicMock:
    """Create a mock LLM that returns the given content string."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content)
    return mock


def test_planner_returns_required_fields(base_state):
    """Planner node should return sub_questions, tools_to_use, and plan_reasoning."""
    from research_analyst.nodes.planner import planner_node

    response_json = """{
        "sub_questions": ["What are Apple's revenue sources?", "What risks does Apple face?"],
        "tools_to_use": ["rag", "finance"],
        "company_name": "Apple",
        "ticker": "AAPL",
        "plan_reasoning": "Using both tools for comprehensive analysis."
    }"""

    with patch("research_analyst.nodes.planner._llm", _mock_llm(response_json)):
        result = planner_node(base_state)

    assert "sub_questions" in result
    assert len(result["sub_questions"]) > 0
    assert "tools_to_use" in result
    assert "plan_reasoning" in result


def test_planner_fallback_on_bad_json(base_state):
    """Planner should fall back gracefully if Claude returns invalid JSON."""
    from research_analyst.nodes.planner import planner_node

    with patch("research_analyst.nodes.planner._llm", _mock_llm("This is not valid JSON")):
        result = planner_node(base_state)

    assert "sub_questions" in result
    assert "tools_to_use" in result


def test_critic_passes_relevant_context(state_with_good_docs):
    """Critic should pass when given relevant documents."""
    from research_analyst.nodes.critic import critic_node

    response_json = """{
        "relevance_score": 8,
        "has_sufficient_context": true,
        "missing_aspects": [],
        "critique": "Documents are highly relevant to the query.",
        "suggested_refinement": "What are Apple's main revenue risks?"
    }"""

    with patch("research_analyst.nodes.critic._llm", _mock_llm(response_json)):
        result = critic_node(state_with_good_docs)

    assert result["critique_result"]["has_sufficient_context"] is True
    assert result["critique_result"]["relevance_score"] >= 7


def test_critic_fails_irrelevant_context(state_with_bad_docs):
    """Critic should fail when given irrelevant documents."""
    from research_analyst.nodes.critic import critic_node

    response_json = """{
        "relevance_score": 2,
        "has_sufficient_context": false,
        "missing_aspects": ["revenue data", "risk factors"],
        "critique": "Retrieved documents are completely unrelated to Apple's financials.",
        "suggested_refinement": "Apple revenue risks iPhone concentration supply chain"
    }"""

    with patch("research_analyst.nodes.critic._llm", _mock_llm(response_json)):
        result = critic_node(state_with_bad_docs)

    assert result["critique_result"]["has_sufficient_context"] is False
    assert result["critique_result"]["relevance_score"] < 7


def test_re_search_plan_increments_retry(base_state):
    """Re-search plan node should increment retry_count and refine the query."""
    from research_analyst.nodes.re_search_plan import re_search_plan_node

    state = {
        **base_state,
        "retry_count": 0,
        "critique_result": {
            "suggested_refinement": "Apple iPhone revenue concentration risk",
            "missing_aspects": ["supply chain", "services growth"],
        },
    }

    result = re_search_plan_node(state)

    assert result["retry_count"] == 1
    assert "Apple iPhone revenue" in result["current_query"]
    assert "web_search" in result["tools_to_use"]


def test_synthesizer_returns_report(state_with_good_docs):
    """Synthesizer should return a non-empty final report."""
    from research_analyst.nodes.synthesizer import synthesizer_node

    report = "## Executive Summary\nApple is a strong company.\n## Conclusion\nBuy."

    with patch("research_analyst.nodes.synthesizer._llm", _mock_llm(report)):
        result = synthesizer_node(state_with_good_docs)

    assert "final_report" in result
    assert len(result["final_report"]) > 0
