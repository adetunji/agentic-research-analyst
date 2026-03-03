# Agentic Research Analyst — Implementation Plan

## Context
Build a portfolio-worthy "Agentic" Research Analyst that demonstrates multi-step LLM reasoning, tool-calling, RAG, and Self-RAG critique loops. The system researches a company by querying internal documents (RAG) and fetching real-time financial data (Yahoo Finance), then critiques its own retrieval quality and retries if context was insufficient.

---

## Recommended Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| LLM | `claude-sonnet-4-6` via `langchain-anthropic` | Best function-calling, native JSON mode, strong analytical reasoning |
| Agent Framework | **LangGraph** | Stateful cyclic graphs; essential for the Self-RAG retry loop |
| Vector DB | **ChromaDB** (local) | Zero signup, persistent, runs on CPU — ideal for portfolio demos |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, fast, 384-dim, strong semantic similarity on financial text |
| Financial Data | **yfinance** | Free, no API key, rich data (price, financials, news, balance sheet) |
| CLI Output | **Rich** | Colored panels, streaming steps — makes portfolio demos visually impressive |
| Testing | `pytest` + `pytest-asyncio` | Standard; handles async LangGraph invocations |

---

## Project Structure

```
agentic_research_analyst/
├── .env                          # ANTHROPIC_API_KEY
├── requirements.txt
├── data/
│   ├── raw/                      # PDFs, TXTs to ingest
│   └── chroma_db/                # Persisted vector store (gitignored)
├── src/research_analyst/
│   ├── config.py                 # Centralized settings (model, paths, thresholds)
│   ├── state.py                  # LangGraph TypedDict state schema
│   ├── graph.py                  # StateGraph assembly (nodes + edges)
│   ├── agent.py                  # Public AgentRunner class
│   ├── tools/
│   │   ├── rag_tool.py           # @tool: search ChromaDB
│   │   ├── finance_tool.py       # @tool: yfinance wrapper
│   │   └── web_search_tool.py    # @tool: DuckDuckGo fallback (optional)
│   ├── vectorstore/
│   │   ├── embeddings.py         # HuggingFaceEmbeddings setup
│   │   ├── store.py              # ChromaDB client
│   │   └── ingest.py             # Load → chunk → enrich metadata → embed
│   └── nodes/
│       ├── planner.py            # Decompose query, select tools
│       ├── tool_executor.py      # Run tools (async parallel)
│       ├── critic.py             # Self-RAG: score retrieved context
│       ├── synthesizer.py        # Generate final research report
│       └── formatter.py          # Rich terminal rendering
├── scripts/
│   ├── ingest_documents.py       # CLI: ingest PDFs into ChromaDB
│   └── run_agent.py              # CLI: query the agent
└── tests/
    ├── conftest.py
    ├── test_tools.py
    ├── test_nodes.py
    └── test_graph.py
```

---

## LangGraph Architecture

```
START → PLANNER → TOOL_EXECUTOR → CRITIC
                       ↑              │
                       │        [FAIL: irrelevant]
                  RE_SEARCH_PLAN ◄────┘

                  CRITIC [PASS] → SYNTHESIZER → FORMATTER → END
```

### Nodes
- **PLANNER**: Decomposes query into sub-questions, picks tools, extracts ticker
- **TOOL_EXECUTOR**: Runs RAG + finance tools concurrently via `asyncio.gather`
- **CRITIC**: Self-RAG — scores retrieved context 0-10, returns JSON with `has_sufficient_context`, `missing_aspects`, `suggested_refinement`
- **RE_SEARCH_PLAN**: On critique FAIL — refines query using critique's `suggested_refinement`, widens `k` from 5 → 10
- **SYNTHESIZER**: Writes structured markdown report (Executive Summary, Financial Highlights, Risks, Sources)
- **FORMATTER**: Renders with Rich panels for demo-ready terminal output

### Conditional Routing (after CRITIC)
```python
def route_after_critic(state) -> str:
    if state["retry_count"] >= MAX_RETRIES:   # default: 2
        return "synthesizer"                  # force forward
    if state["critique_result"]["has_sufficient_context"]:
        return "synthesizer"
    return "re_search_plan"
```

---

## State Schema (`state.py`)

```python
class ResearchState(TypedDict):
    query: str
    company_name: str
    ticker: str
    sub_questions: list[str]
    tools_to_use: list[str]
    # Annotated with operator.add → retries APPEND, not overwrite
    retrieved_documents: Annotated[list[Document], operator.add]
    financial_data: dict[str, Any]
    critique_result: CritiqueResult
    critique_history: Annotated[list[CritiqueResult], operator.add]
    retry_count: int
    final_report: str
    messages: list[dict]
    error: str | None
```

---

## Self-RAG Critique Prompt

The CRITIC node sends to Claude:
```
Evaluate whether retrieved documents sufficiently answer the query.
Return JSON: { relevance_score (0-10), has_sufficient_context (bool),
               missing_aspects (list), critique (str), suggested_refinement (str) }
Score >= 7 AND has_sufficient_context = true → PASS
```

---

## Key Dependencies (`requirements.txt`)

```
langgraph==0.2.50
langchain-anthropic==0.3.4
anthropic==0.46.0
chromadb==0.6.3
langchain-chroma==0.2.2
sentence-transformers==3.4.1
langchain-huggingface==0.1.2
langchain-community==0.3.15
pypdf==5.3.1
yfinance==0.2.54
pandas==2.2.3
python-dotenv==1.0.1
pydantic==2.10.6
rich==13.9.4
pytest==8.3.4
pytest-asyncio==0.25.3
duckduckgo-search==6.3.7
```

---

## Build Order

1. `requirements.txt` + `.env.example` + `pyproject.toml`
2. `state.py` — all other files depend on this
3. `config.py` — settings loaded by everything
4. `vectorstore/` — embeddings, store, ingest
5. `tools/` — rag_tool, finance_tool
6. `nodes/` — planner → tool_executor → critic → re_search_plan → synthesizer → formatter
7. `graph.py` — wire all nodes + edges
8. `agent.py` — wrapper class
9. `scripts/` — ingest_documents.py, run_agent.py
10. `tests/`

---

## Todos

- [x] Create project scaffolding (requirements.txt, .env.example, .gitignore, pyproject.toml, directories) ✓
- [x] Create `state.py` — ResearchState TypedDict schema ✓
- [x] Create `config.py` — centralized settings ✓
- [x] Create `vectorstore/` — embeddings, store, ingest pipeline ✓
- [x] Create `tools/` — rag_tool, finance_tool, web_search_tool ✓
- [x] Create `nodes/` — planner, tool_executor, critic, re_search_plan, synthesizer, formatter ✓
- [x] Create `graph.py` — LangGraph StateGraph assembly
- [x] Create `agent.py` — public AgentRunner class
- [x] Create `scripts/` — ingest_documents.py and run_agent.py CLI scripts ✓
- [x] Create `tests/` — conftest, test_tools, test_nodes, test_graph ✓

---

## Verification

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Ingest sample docs
python scripts/ingest_documents.py --dir data/raw --company Apple --ticker AAPL

# 3. Verify ChromaDB
python -c "import chromadb; c=chromadb.PersistentClient('data/chroma_db'); print(c.get_collection('research_documents').count())"

# 4. Run unit tests
pytest tests/ -v

# 5. Full end-to-end
python scripts/run_agent.py \
  --query "Analyze Apple's financial health and key risks" \
  --company "Apple" --ticker "AAPL"
```

**What to verify in E2E run:**
- Planner shows sub-questions + tool selection
- CRITIC shows relevance score (7+ with good docs)
- If score < 7: RE_SEARCH_PLAN fires, second retrieval appends
- Final report has sections: Executive Summary, Financial Highlights, Risk Factors, Sources
- No hallucinated source names — all cited docs exist in `data/raw/`
