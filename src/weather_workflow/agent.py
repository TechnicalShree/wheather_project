import json

from .config import MODEL_NAME, client
from .schemas import ConceptExplanation
from .tools import execute_tool, weather_tool_definition


def run_basic_completion(prompt: str, system_prompt: str = "You are a helpful AI instructor. Explain concepts simply.") -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=300,
    )
    return response.choices[0].message.content


def run_json_mode_completion(prompt: str, system_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"Return only a valid JSON object. {system_prompt}"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def run_structured_output_completion(prompt: str, system_prompt: str) -> ConceptExplanation:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "concept_explanation",
                "strict": True,
                "schema": ConceptExplanation.model_json_schema(),
            },
        },
        temperature=0,
    )
    return ConceptExplanation.model_validate_json(response.choices[0].message.content)


def ask_with_weather_tool(question: str, max_rounds: int = 5) -> str:
    messages = [
        {"role": "system", "content": "Use the weather tool for current weather questions."},
        {"role": "user", "content": question},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[weather_tool_definition],
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            return assistant_message.content

        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            print("Tool called:", tool_call.function.name)
            print("Arguments:", tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, tool_call.function.arguments)
            print("Tool result:", result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError("Tool-calling loop exceeded max_rounds without a final answer")
