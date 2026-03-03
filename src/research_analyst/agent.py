from research_analyst.graph import build_graph
from research_analyst.state import ResearchState


class AgentRunner:
    """
    Public interface for the research agent.
    Wraps the compiled LangGraph and provides a clean run() method.
    """

    def __init__(self):
        self.graph = build_graph()

    def run(self, query: str, company_name: str = "", ticker: str = "") -> ResearchState:
        """
        Run the full research agent pipeline for a given query.

        Args:
            query:        The research question, e.g. "Analyze Apple's key risks"
            company_name: Company name, e.g. "Apple"
            ticker:       Stock ticker symbol, e.g. "AAPL"

        Returns:
            The final ResearchState containing the report and all intermediate results.
        """
        initial_state: ResearchState = {
            "query": query,
            "company_name": company_name,
            "ticker": ticker,
            "sub_questions": [],
            "tools_to_use": [],
            "plan_reasoning": "",
            "retrieved_documents": [],
            "financial_data": {},
            "current_query": query,
            "critique_result": {},
            "critique_history": [],
            "retry_count": 0,
            "final_report": "",
            "formatted_output": "",
            "messages": [],
            "error": None,
        }

        return self.graph.invoke(initial_state)
