from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WebSearchToolInput(BaseModel):
    query: str = Field(description="Search query for recent news or information")


@tool("web_search", args_schema=WebSearchToolInput)
def web_search(query: str) -> str:
    """
    Search the web for recent news, analyst opinions, or information not
    available in internal documents or Yahoo Finance. Use as a last resort
    when internal docs and financial data are insufficient.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No web search results found."

        lines = [f"**Web Search Results for:** {query}"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"\n[{i}] {title}\n{body}\n{href}")

        return "\n".join(lines)

    except Exception as e:
        return f"Web search failed: {e}"
