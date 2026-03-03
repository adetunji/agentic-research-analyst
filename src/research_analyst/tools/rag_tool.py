from langchain_core.tools import tool
from pydantic import BaseModel, Field

from research_analyst.config import settings
from research_analyst.vectorstore.store import get_retriever


class RAGToolInput(BaseModel):
    query: str = Field(description="Search query to find relevant internal documents")
    company_name: str = Field(default="", description="Company name to filter results")
    k: int = Field(default=5, description="Number of documents to retrieve")


@tool("search_internal_documents", args_schema=RAGToolInput)
def search_internal_documents(query: str, company_name: str = "", k: int = 5) -> str:
    """
    Search internal documents, research notes, annual reports, and filings
    stored in the vector database. Use this for qualitative analysis, historical
    context, strategy details, risk factors, and internal assessments.
    """
    retriever = get_retriever(k=k)

    # Filter by company if provided
    if company_name:
        retriever.search_kwargs["filter"] = {"company": company_name}

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found in the internal knowledge base."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        formatted.append(
            f"[Document {i}] Source: {source}{page_str}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)
