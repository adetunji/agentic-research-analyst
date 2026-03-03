import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def test_rag_tool_returns_string_on_empty_db():
    """RAG tool should return a fallback message when no docs are found."""
    from research_analyst.tools.rag_tool import search_internal_documents

    with patch("research_analyst.tools.rag_tool.get_retriever") as mock_retriever:
        mock_retriever.return_value.invoke.return_value = []
        result = search_internal_documents.invoke({
            "query": "Apple revenue risks",
            "company_name": "Apple",
            "k": 5,
        })

    assert isinstance(result, str)
    assert "No relevant documents" in result


def test_rag_tool_formats_results():
    """RAG tool should format documents with source metadata."""
    from research_analyst.tools.rag_tool import search_internal_documents

    mock_doc = Document(
        page_content="Apple revenue depends heavily on iPhone sales.",
        metadata={"source": "apple_10k.pdf", "page": 5},
    )

    with patch("research_analyst.tools.rag_tool.get_retriever") as mock_retriever:
        mock_retriever.return_value.invoke.return_value = [mock_doc]
        result = search_internal_documents.invoke({
            "query": "Apple revenue",
            "company_name": "Apple",
            "k": 3,
        })

    assert "apple_10k.pdf" in result
    assert "Apple revenue depends heavily" in result


def test_finance_tool_overview_returns_string():
    """Finance tool should return a formatted string for overview data type."""
    from research_analyst.tools.finance_tool import get_financial_data

    with patch("research_analyst.tools.finance_tool.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {
            "longName": "Apple Inc.",
            "marketCap": 2900000000000,
            "currentPrice": 189.5,
            "trailingPE": 29.4,
        }
        result = get_financial_data.invoke({"ticker": "AAPL", "data_type": "overview"})

    assert isinstance(result, str)
    assert "AAPL" in result


def test_finance_tool_handles_bad_ticker():
    """Finance tool should return an error string for invalid tickers."""
    from research_analyst.tools.finance_tool import get_financial_data

    with patch("research_analyst.tools.finance_tool.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.info = {}
        result = get_financial_data.invoke({"ticker": "INVALID999", "data_type": "overview"})

    assert isinstance(result, str)
    assert "No data found" in result or "Error" in result


def test_web_search_tool_returns_string():
    """Web search tool should return a formatted string of results."""
    from research_analyst.tools.web_search_tool import web_search

    # DDGS is imported inside the function body, so patch at the source package
    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Apple Q4 Results", "body": "Apple reported record revenue.", "href": "http://example.com"}
        ]
        result = web_search.invoke({"query": "Apple earnings 2024"})

    assert isinstance(result, str)
    assert "Apple Q4 Results" in result
