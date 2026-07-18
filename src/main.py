"""Runs all four workflows: basic call, JSON mode, structured output, tool calling."""
import json

from pydantic import ValidationError

from .modules.llm import basic_completion, json_completion, structured_completion
from .modules.weather import ask_with_weather_tool


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    print_header("1. Basic OpenRouter API Call")
    print(basic_completion("Explain what an API is in two sentences."))

    print_header("2. JSON Mode")
    data = json_completion(
        "Explain cosine similarity with two simple examples.",
        keys_hint="Include the keys topic, explanation, and examples.",
    )
    print(json.dumps(data, indent=2))

    print_header("3. Structured Output & Pydantic Validation")
    try:
        concept = structured_completion("Explain LoRA scaling in simple language.")
        print("✅ Response passed Pydantic validation")
        print(concept.model_dump_json(indent=2))
    except ValidationError as err:
        print("❌ Response failed Pydantic validation!")
        print(err)

    print_header("4. Tool Calling Weather Workflow")
    print(ask_with_weather_tool("What is the current weather in Kolkata?"))
    print()
    print(ask_with_weather_tool("Why does humidity sometimes make hot weather feel more uncomfortable?"))


if __name__ == "__main__":
    main()
