from research_analyst.state import ResearchState
from research_analyst.tools.rag_tool import search_internal_documents
from research_analyst.tools.finance_tool import get_financial_data
from research_analyst.tools.web_search_tool import web_search
from langchain_core.documents import Document


def tool_executor_node(state: ResearchState) -> dict:
    """
    Executes the tools selected by the planner.
    RAG results are stored as Documents; financial data as a dict.
    """
    tools_to_use = state.get("tools_to_use", ["rag", "finance"])
    query = state.get("current_query", state["query"])
    company_name = state.get("company_name", "")
    ticker = state.get("ticker", "")
    retry_count = state.get("retry_count", 0)

    # Widen search on retries
    k = settings_k(retry_count)

    retrieved_documents = []
    financial_data = state.get("financial_data") or {}

    if "rag" in tools_to_use:
        raw = search_internal_documents.invoke({
            "query": query,
            "company_name": company_name,
            "k": k,
        })
        # Wrap the string result as a Document for consistent state typing
        retrieved_documents.append(
            Document(page_content=raw, metadata={"source": "internal_rag", "query": query})
        )

    if "finance" in tools_to_use and ticker:
        for data_type in ["overview", "financials", "news"]:
            result = get_financial_data.invoke({
                "ticker": ticker,
                "data_type": data_type,
            })
            financial_data[data_type] = result

    if "web_search" in tools_to_use:
        raw = web_search.invoke({"query": query})
        retrieved_documents.append(
            Document(page_content=raw, metadata={"source": "web_search", "query": query})
        )

    return {
        "retrieved_documents": retrieved_documents,
        "financial_data": financial_data,
        "retry_count": retry_count,
    }


def settings_k(retry_count: int) -> int:
    """Return wider k on retries."""
    from research_analyst.config import settings
    return settings.rag_retry_top_k if retry_count > 0 else settings.rag_top_k
