import os
from typing import Any

import requests

from .schemas import WeatherToolInput


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


weather_tool_definition = {
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

AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather,
}


def execute_tool(tool_name: str, raw_arguments: str) -> dict[str, Any]:
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool requested: {tool_name}")

    validated_args = WeatherToolInput.model_validate_json(raw_arguments)
    return AVAILABLE_TOOLS[tool_name](**validated_args.model_dump())
