"""Weather tool: OpenWeather call, tool schema, and the tool-calling agent loop."""
import json
import os
from typing import Any

import requests
from pydantic import BaseModel, Field

from .llm import MODEL_NAME, client


class WeatherToolInput(BaseModel):
    location: str = Field(description="Name of the city, for example Kolkata or Paris")


def get_current_weather(location: str) -> dict[str, Any]:
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": location,
            "appid": os.environ["OPENWEATHER_API_KEY"],
            "units": "metric",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "location": data["name"],
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "condition": data["weather"][0]["description"],
    }


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current temperature, humidity, wind speed, "
            "and weather condition for a city using OpenWeather."
        ),
        "parameters": WeatherToolInput.model_json_schema(),
    },
}


def ask_with_weather_tool(question: str, max_rounds: int = 5) -> str:
    """User question → LLM picks tool → run tool → result back to LLM → answer."""
    messages = [
        {"role": "system", "content": "Use the weather tool for current weather questions."},
        {"role": "user", "content": question},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            args = WeatherToolInput.model_validate_json(tool_call.function.arguments)
            print("Tool called:", tool_call.function.name, "| Arguments:", args.model_dump())
            result = get_current_weather(args.location)
            print("Tool result:", result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    raise RuntimeError("Tool-calling loop exceeded max_rounds without a final answer")
