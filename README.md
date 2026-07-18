# Weather Tool-Calling Workflow

OpenRouter workflows demonstrating: basic chat completion, JSON mode, structured output with Pydantic validation, and a weather tool-calling agent loop (OpenWeather).

## Structure

```
src/weather_workflow/
├── config.py    # OpenRouter client + model name
├── schemas.py   # Pydantic models (ConceptExplanation, WeatherToolInput)
├── tools.py     # get_current_weather + tool definition + router
├── agent.py     # LLM call helpers + tool-calling loop
└── main.py      # Runs all four steps
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

## Run

```bash
python -m src.weather_workflow.main
```
