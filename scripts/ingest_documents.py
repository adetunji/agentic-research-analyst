#!/usr/bin/env python
"""
CLI script to ingest documents into ChromaDB.

Usage:
    python scripts/ingest_documents.py \
        --dir data/raw \
        --company Apple \
        --ticker AAPL \
        --doc-type annual_report
"""
import argparse
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_analyst.vectorstore.ingest import ingest_documents


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--dir", required=True, help="Directory containing PDF/TXT files")
    parser.add_argument("--company", required=True, help="Company name, e.g. Apple")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--doc-type", default="report", help="Document type label (default: report)")
    args = parser.parse_args()

    count = ingest_documents(
        data_dir=args.dir,
        company_name=args.company,
        ticker=args.ticker,
        doc_type=args.doc_type,
    )

    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
