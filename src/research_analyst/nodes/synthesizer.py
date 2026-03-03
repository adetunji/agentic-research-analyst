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

_SYSTEM_PROMPT = """You are a senior equity research analyst. Write a structured research report
based on the provided internal documents and financial data.

Your report MUST follow this structure:

## Executive Summary
2-3 sentence overview of the company's current position.

## Financial Highlights
Key metrics and trends from the financial data. Use specific numbers.

## Strategic Analysis
Insights from internal documents on strategy, competitive position, and business model.

## Risk Factors
Key risks identified from both internal documents and financial data.

## Conclusion
Investment outlook and key takeaways.

## Sources
List every document and data source used, with page numbers where available.

Rules:
- Cite specific sources inline, e.g. (Apple 10-K 2024, p.12)
- Flag explicitly where data is missing or uncertain
- Do NOT hallucinate source names — only cite sources present in the context
- Be concise but substantive — this is a professional report"""


def _format_all_context(state: ResearchState) -> str:
    docs = state.get("retrieved_documents", [])
    financial_data = state.get("financial_data", {})
    sub_questions = state.get("sub_questions", [])

    sections = []

    if sub_questions:
        sections.append("RESEARCH QUESTIONS:\n" + "\n".join(f"- {q}" for q in sub_questions))

    if docs:
        doc_texts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            doc_texts.append(f"[Document {i} | Source: {source}]\n{doc.page_content}")
        sections.append("INTERNAL DOCUMENTS:\n\n" + "\n\n---\n\n".join(doc_texts))

    if financial_data:
        fin_parts = []
        for key, value in financial_data.items():
            fin_parts.append(f"[{key.upper()}]\n{value}")
        sections.append("FINANCIAL DATA:\n\n" + "\n\n".join(fin_parts))

    return "\n\n" + ("=" * 60) + "\n\n".join(sections)


def synthesizer_node(state: ResearchState) -> dict:
    """
    Generates the final structured research report using all retrieved
    documents and financial data.
    """
    query = state["query"]
    company = state.get("company_name", "the company")
    context = _format_all_context(state)

    user_content = f"""Write a research report for: {company}

ORIGINAL QUERY: {query}

CONTEXT:
{context}"""

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    return {"final_report": response.content}
