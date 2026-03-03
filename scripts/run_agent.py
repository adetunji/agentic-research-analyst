#!/usr/bin/env python
"""
CLI entry point for the Agentic Research Analyst.

Usage:
    python scripts/run_agent.py \
        --query "Analyze Apple's financial health and key risks" \
        --company "Apple" \
        --ticker "AAPL"
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_analyst.agent import AgentRunner

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run the Agentic Research Analyst")
    parser.add_argument("--query", required=True, help="Research question to answer")
    parser.add_argument("--company", default="", help="Company name, e.g. Apple")
    parser.add_argument("--ticker", default="", help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    console.print(Rule("[bold cyan] Agentic Research Analyst [/bold cyan]"))
    console.print(f"[cyan]Query:[/cyan] {args.query}")
    if args.company:
        console.print(f"[cyan]Company:[/cyan] {args.company} ({args.ticker})")
    console.print()

    runner = AgentRunner()

    try:
        runner.run(
            query=args.query,
            company_name=args.company,
            ticker=args.ticker,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
