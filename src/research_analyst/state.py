from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.documents import Document


class CritiqueResult(TypedDict):
    relevance_score: int           # 0-10
    has_sufficient_context: bool
    missing_aspects: list[str]
    critique: str
    suggested_refinement: str


class ResearchState(TypedDict):
    # --- Input ---
    query: str                     # Original user question
    company_name: str              # e.g. "Apple"
    ticker: str                    # e.g. "AAPL"

    # --- Planning ---
    sub_questions: list[str]       # Decomposed research questions
    tools_to_use: list[str]        # e.g. ["rag", "finance"]
    plan_reasoning: str            # Planner's chain-of-thought

    # --- Retrieval ---
    # operator.add means retries APPEND rather than overwrite
    retrieved_documents: Annotated[list[Document], operator.add]
    financial_data: dict[str, Any]
    current_query: str             # Potentially refined query for current iteration

    # --- Self-RAG ---
    critique_result: CritiqueResult
    critique_history: Annotated[list[CritiqueResult], operator.add]
    retry_count: int

    # --- Output ---
    final_report: str              # Markdown research report
    formatted_output: str          # Rich-rendered terminal output

    # --- Metadata ---
    messages: list[dict]           # LangGraph message history
    error: str | None
