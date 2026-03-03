import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def _mock_llm(content: str) -> MagicMock:
    """Create a mock LLM that returns the given content string."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=content)
    return mock


def test_route_after_critic_pass():
    """Should route to synthesizer when critique passes."""
    from research_analyst.graph import route_after_critic

    state = {
        "retry_count": 0,
        "critique_result": {"has_sufficient_context": True, "relevance_score": 8},
    }
    assert route_after_critic(state) == "synthesizer"


def test_route_after_critic_fail():
    """Should route to re_search_plan when critique fails and retries remain."""
    from research_analyst.graph import route_after_critic

    state = {
        "retry_count": 0,
        "critique_result": {"has_sufficient_context": False, "relevance_score": 3},
    }
    assert route_after_critic(state) == "re_search_plan"


def test_route_after_critic_max_retries():
    """Should force synthesizer when max retries are exhausted."""
    from research_analyst.graph import route_after_critic

    state = {
        "retry_count": 2,  # equals max_critique_retries default
        "critique_result": {"has_sufficient_context": False, "relevance_score": 2},
    }
    assert route_after_critic(state) == "synthesizer"


def test_full_graph_happy_path():
    """
    Integration test: graph should complete and produce a final report
    when all nodes succeed on the first pass.
    """
    from research_analyst.graph import build_graph

    planner_json = """{
        "sub_questions": ["What are Apple's risks?"],
        "tools_to_use": ["finance"],
        "company_name": "Apple",
        "ticker": "AAPL",
        "plan_reasoning": "Finance only — purely quantitative query."
    }"""

    critic_json = """{
        "relevance_score": 9,
        "has_sufficient_context": true,
        "missing_aspects": [],
        "critique": "Financial data is highly relevant.",
        "suggested_refinement": "Apple revenue risks"
    }"""

    with patch("research_analyst.nodes.planner._llm", _mock_llm(planner_json)), \
         patch("research_analyst.nodes.critic._llm", _mock_llm(critic_json)), \
         patch("research_analyst.nodes.synthesizer._llm",
               _mock_llm("## Executive Summary\nApple looks strong.")), \
         patch("research_analyst.tools.finance_tool.yf.Ticker") as mock_ticker, \
         patch("research_analyst.tools.rag_tool.get_retriever") as mock_retriever:

        mock_ticker.return_value.info = {"longName": "Apple Inc.", "marketCap": 2900000000000}
        mock_ticker.return_value.financials = MagicMock()
        mock_ticker.return_value.balance_sheet = MagicMock()
        mock_ticker.return_value.news = []
        mock_retriever.return_value.invoke.return_value = []

        graph = build_graph()
        result = graph.invoke({
            "query": "Analyze Apple's financial health",
            "company_name": "Apple",
            "ticker": "AAPL",
            "sub_questions": [],
            "tools_to_use": [],
            "plan_reasoning": "",
            "retrieved_documents": [],
            "financial_data": {},
            "current_query": "Analyze Apple's financial health",
            "critique_result": {},
            "critique_history": [],
            "retry_count": 0,
            "final_report": "",
            "formatted_output": "",
            "messages": [],
            "error": None,
        })

    assert result["final_report"] != ""
    assert "Apple" in result["final_report"]
