# Weather Tool-Calling Workflow

OpenRouter workflows: basic chat completion, JSON mode, structured output with Pydantic validation, and a weather tool-calling agent loop (OpenWeather).

## Structure

```
src/
├── modules/
│   ├── llm.py       # OpenRouter client + basic/JSON/structured completion helpers
│   └── weather.py   # Weather tool, its schema, and the tool-calling loop
└── main.py          # Runs all four workflows
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

## Run

```bash
python -m src.main
```
