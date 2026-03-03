import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from research_analyst.config import settings
from research_analyst.state import ResearchState, CritiqueResult

_llm = ChatAnthropic(
    model=settings.model_name,
    temperature=0.0,  # Zero temp for consistent scoring
    max_tokens=1024,
    api_key=settings.anthropic_api_key,
)

_SYSTEM_PROMPT = """You are a research quality evaluator. Your job is to assess whether
retrieved context is sufficient to answer a research query about a company.

Respond ONLY with valid JSON in this exact format:
{
  "relevance_score": <int 0-10>,
  "has_sufficient_context": <bool>,
  "missing_aspects": ["aspect 1", "aspect 2"],
  "critique": "<one paragraph explanation>",
  "suggested_refinement": "<rewritten query to find missing info>"
}

Scoring rules:
- 8-10: Highly relevant, covers the query well
- 5-7: Partially relevant, some gaps
- 0-4: Largely irrelevant or empty

Set has_sufficient_context = true only if relevance_score >= 7 AND the context
meaningfully addresses the query. If financial data is present but internal docs
are empty, that may still pass for purely financial queries."""


def _format_docs(state: ResearchState) -> str:
    docs = state.get("retrieved_documents", [])
    if not docs:
        return "No documents retrieved."
    return "\n\n---\n\n".join(
        f"[Doc {i+1}]\n{doc.page_content[:800]}" for i, doc in enumerate(docs)
    )


def _format_financials(state: ResearchState) -> str:
    financial_data = state.get("financial_data", {})
    if not financial_data:
        return "No financial data retrieved."
    parts = []
    for key, value in financial_data.items():
        parts.append(f"[{key.upper()}]\n{str(value)[:500]}")
    return "\n\n".join(parts)


def critic_node(state: ResearchState) -> dict:
    """
    Self-RAG critique node. Scores the retrieved context and decides
    whether to proceed to synthesis or trigger a re-search.
    """
    query = state.get("current_query", state["query"])
    company = state.get("company_name", "")
    docs_text = _format_docs(state)
    financials_text = _format_financials(state)

    user_content = f"""ORIGINAL QUERY: {query}
COMPANY: {company}

RETRIEVED DOCUMENTS:
{docs_text}

FINANCIAL DATA:
{financials_text}"""

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    try:
        result: CritiqueResult = json.loads(response.content)
    except json.JSONDecodeError:
        # Default to PASS on parse failure to avoid infinite loops
        result = CritiqueResult(
            relevance_score=7,
            has_sufficient_context=True,
            missing_aspects=[],
            critique="Could not parse critique response — defaulting to pass.",
            suggested_refinement=query,
        )

    return {
        "critique_result": result,
        "critique_history": [result],
        "retry_count": state.get("retry_count", 0),
    }
