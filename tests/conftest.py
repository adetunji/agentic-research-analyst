import pytest
from langchain_core.documents import Document
from research_analyst.state import ResearchState, CritiqueResult


@pytest.fixture
def base_state() -> ResearchState:
    return {
        "query": "What are Apple's main revenue risks?",
        "company_name": "Apple",
        "ticker": "AAPL",
        "sub_questions": [],
        "tools_to_use": ["rag", "finance"],
        "plan_reasoning": "",
        "retrieved_documents": [],
        "financial_data": {},
        "current_query": "What are Apple's main revenue risks?",
        "critique_result": {},
        "critique_history": [],
        "retry_count": 0,
        "final_report": "",
        "formatted_output": "",
        "messages": [],
        "error": None,
    }


@pytest.fixture
def state_with_good_docs(base_state) -> ResearchState:
    """State with relevant retrieved documents — should pass critique."""
    return {
        **base_state,
        "retrieved_documents": [
            Document(
                page_content="Apple's primary revenue risk stems from iPhone concentration. "
                             "Services segment growth partially offsets hardware dependency. "
                             "Supply chain disruptions in Asia remain a key concern.",
                metadata={"source": "apple_10k_2024.pdf", "page": 12, "company": "Apple"},
            )
        ],
        "financial_data": {
            "overview": "currentPrice: 189.5\nmarketCap: 2900000000000\ntrailingPE: 29.4"
        },
    }


@pytest.fixture
def state_with_bad_docs(base_state) -> ResearchState:
    """State with irrelevant retrieved documents — should fail critique."""
    return {
        **base_state,
        "retrieved_documents": [
            Document(
                page_content="The weather in California was sunny last Tuesday.",
                metadata={"source": "unrelated.txt", "company": "Apple"},
            )
        ],
        "financial_data": {},
    }
