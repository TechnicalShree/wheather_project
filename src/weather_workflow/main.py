import json

from pydantic import ValidationError

from .agent import (
    ask_with_weather_tool,
    run_basic_completion,
    run_json_mode_completion,
    run_structured_output_completion,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> None:
    # Step 1: Basic completion
    print_header("1. Basic OpenRouter API Call")
    prompt = "Explain what an API is in two sentences."
    print(f"Prompt: {prompt}\n")
    print(run_basic_completion(prompt))

    # Step 2: JSON mode
    print_header("2. JSON Mode")
    prompt = "Explain cosine similarity with two simple examples."
    print(f"Prompt: {prompt}\n")
    json_data = run_json_mode_completion(
        prompt=prompt,
        system_prompt="Include the keys topic, explanation, and examples.",
    )
    print("Parsed JSON Response:")
    print(json.dumps(json_data, indent=2))

    # Step 3: Structured output with Pydantic validation
    print_header("3. Structured Output & Pydantic Validation")
    prompt = "Explain LoRA scaling in simple language."
    print(f"Prompt: {prompt}\n")
    try:
        concept = run_structured_output_completion(
            prompt=prompt,
            system_prompt="You are an AI instructor. Follow the supplied JSON schema exactly.",
        )
        print("✅ Response passed Pydantic validation")
        print(concept.model_dump_json(indent=2))
    except ValidationError as val_err:
        print("❌ Response failed Pydantic validation!")
        print(val_err)

    # Step 4: Weather tool-calling workflow
    print_header("4. Tool Calling Weather Workflow")

    question_a = "What is the current weather in Kolkata?"
    print(f"Question A: {question_a}\n")
    print(ask_with_weather_tool(question_a))

    question_b = "Why does humidity sometimes make hot weather feel more uncomfortable?"
    print(f"\nQuestion B: {question_b}\n")
    print(ask_with_weather_tool(question_b))


if __name__ == "__main__":
    main()
