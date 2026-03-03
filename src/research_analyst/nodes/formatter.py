from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from research_analyst.state import ResearchState

console = Console()


def formatter_node(state: ResearchState) -> dict:
    """
    Renders the final research report and agent reasoning to the terminal
    using Rich for a polished, demo-ready output.
    """
    company = state.get("company_name", "Company")
    ticker = state.get("ticker", "")
    plan_reasoning = state.get("plan_reasoning", "")
    critique_history = state.get("critique_history", [])
    retry_count = state.get("retry_count", 0)
    final_report = state.get("final_report", "No report generated.")

    # --- Planning summary ---
    console.print(Rule(f"[bold cyan] Research Agent: {company} ({ticker}) [/bold cyan]"))

    if plan_reasoning:
        console.print(Panel(plan_reasoning, title="[cyan]Planner Reasoning[/cyan]", expand=False))

    # --- Critique history ---
    for i, critique in enumerate(critique_history, 1):
        score = critique.get("relevance_score", "?")
        passed = critique.get("has_sufficient_context", False)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        critique_text = critique.get("critique", "")
        console.print(
            Panel(
                f"Score: {score}/10 | Status: {status}\n\n{critique_text}",
                title=f"[yellow]Critic — Round {i}[/yellow]",
                expand=False,
            )
        )

    if retry_count > 0:
        console.print(f"[yellow]Re-searched {retry_count} time(s) to improve context.[/yellow]")

    # --- Final report ---
    console.print(Rule("[bold green] Final Research Report [/bold green]"))
    console.print(Markdown(final_report))
    console.print(Rule())

    return {"formatted_output": final_report}
