import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from research_analyst.config import settings
from research_analyst.state import ResearchState

_llm = ChatAnthropic(
    model=settings.model_name,
    temperature=settings.temperature,
    max_tokens=settings.max_tokens,
    api_key=settings.anthropic_api_key,
)

_SYSTEM_PROMPT = """You are a research planning assistant. Given a user query about a company,
your job is to:
1. Decompose the query into 2-4 focused sub-questions
2. Decide which tools to use: "rag" (internal documents), "finance" (Yahoo Finance), or both
3. Extract the company name and ticker symbol if not already provided

Respond ONLY with valid JSON in this exact format:
{
  "sub_questions": ["question 1", "question 2", ...],
  "tools_to_use": ["rag", "finance"],
  "company_name": "Apple",
  "ticker": "AAPL",
  "plan_reasoning": "Brief explanation of your plan"
}

Rules:
- Always include "finance" if the query involves numbers, prices, ratios, or recent performance
- Always include "rag" if the query involves strategy, risks, qualitative analysis, or history
- Include both for general company analysis queries
- ticker should be uppercase, e.g. AAPL not aapl"""


def planner_node(state: ResearchState) -> dict:
    """
    Decomposes the user query into sub-questions and selects which tools to use.
    Also extracts company name and ticker if not already in state.
    """
    query = state["query"]
    company_name = state.get("company_name", "")
    ticker = state.get("ticker", "")

    user_content = f"Query: {query}"
    if company_name:
        user_content += f"\nCompany: {company_name}"
    if ticker:
        user_content += f"\nTicker: {ticker}"

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    try:
        plan = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback if Claude returns non-JSON
        plan = {
            "sub_questions": [query],
            "tools_to_use": ["rag", "finance"],
            "company_name": company_name,
            "ticker": ticker,
            "plan_reasoning": "Fallback plan: using all tools.",
        }

    return {
        "sub_questions": plan.get("sub_questions", [query]),
        "tools_to_use": plan.get("tools_to_use", ["rag", "finance"]),
        "company_name": plan.get("company_name", company_name),
        "ticker": plan.get("ticker", ticker),
        "plan_reasoning": plan.get("plan_reasoning", ""),
        "current_query": query,
    }
