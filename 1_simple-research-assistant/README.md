# Simple Research Assistant

A LangGraph-based research assistant that uses Groq LLM and external tools for:
- Web search (Tavily)
- Backup search (DuckDuckGo)
- Live stock price lookup (Alpha Vantage)

## Project Structure

- `main.py` — graph, tools, and execution entry point
- `test.ipynb` — experimentation notebook
- `workflow.png` — auto-generated graph visualization

## Prerequisites

- Python 3.10+
- API keys:
  - `GROQ_API_KEY`
  - `TAVILY_API_KEY`
  - `ALPHA_VANTAGE_API_KEY` (optional if default key in code is used)

## Setup

From workspace root (`langgraph-agents`):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create or update `.env` in workspace root:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

## Run

From `1_simple-research-assistant` folder:

```powershell
python main.py
```

The script will:
1. Build and save graph image to `workflow.png`
2. Invoke the chatbot with the sample prompt in `main.py`
3. Print model output

## Tools Configured

- `tavily_search(query: str)`
- `duckduckgo_search(query: str)`
- `get_stock_price(symbol: str)`

 