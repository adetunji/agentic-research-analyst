import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class FinanceToolInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL, TSLA, NVDA")
    data_type: str = Field(
        default="overview",
        description="Type of data to fetch: 'overview', 'financials', 'history', 'news'",
    )
    period: str = Field(
        default="1y",
        description="Time period for history: '1mo', '3mo', '6mo', '1y', '5y'",
    )


def _format_dataframe(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Serialize a DataFrame to a readable markdown-style string."""
    if df is None or df.empty:
        return "No data available."
    return df.head(max_rows).to_string()


@tool("get_financial_data", args_schema=FinanceToolInput)
def get_financial_data(ticker: str, data_type: str = "overview", period: str = "1y") -> str:
    """
    Fetch real-time and historical financial data from Yahoo Finance.
    Use for current price, market cap, P/E ratio, revenue, earnings,
    balance sheet, recent news, and price history. Requires a valid ticker symbol.
    """
    try:
        stock = yf.Ticker(ticker)

        if data_type == "overview":
            info = stock.info
            if not info:
                return f"No data found for ticker '{ticker}'."
            keys = [
                "longName", "sector", "industry", "marketCap", "currentPrice",
                "trailingPE", "forwardPE", "priceToBook", "revenueGrowth",
                "grossMargins", "operatingMargins", "profitMargins",
                "totalRevenue", "netIncomeToCommon", "totalDebt",
                "totalCash", "returnOnEquity", "beta", "52WeekChange",
            ]
            lines = [f"**{ticker.upper()} Overview**"]
            for key in keys:
                value = info.get(key)
                if value is not None:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)

        elif data_type == "financials":
            income = stock.financials
            balance = stock.balance_sheet
            result = f"**{ticker.upper()} Income Statement**\n{_format_dataframe(income)}"
            result += f"\n\n**{ticker.upper()} Balance Sheet**\n{_format_dataframe(balance)}"
            return result

        elif data_type == "history":
            hist = stock.history(period=period)
            if hist.empty:
                return f"No price history found for '{ticker}' over period '{period}'."
            summary = hist[["Open", "High", "Low", "Close", "Volume"]].tail(10)
            return f"**{ticker.upper()} Price History (last 10 periods of {period})**\n{summary.to_string()}"

        elif data_type == "news":
            news = stock.news
            if not news:
                return f"No recent news found for '{ticker}'."
            lines = [f"**{ticker.upper()} Recent News**"]
            for item in news[:8]:
                title = item.get("title", "No title")
                link = item.get("link", "")
                lines.append(f"- {title}\n  {link}")
            return "\n".join(lines)

        else:
            return f"Unknown data_type '{data_type}'. Use: overview, financials, history, or news."

    except Exception as e:
        return f"Error fetching financial data for '{ticker}': {e}"
